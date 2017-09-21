/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.feature

import scala.collection.mutable.ArrayBuilder
import org.json4s._
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._
import breeze.linalg.{ DenseVector => BDV, SparseVector => BSV, Vector => BV, DenseMatrix => BDM }
import org.apache.spark.annotation.{ Since, Experimental }
import org.apache.spark.mllib.feature.{ InfoThCriterionFactory => FT }
import org.apache.spark.mllib.feature.{ InfoTheory => IT }
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.{ Vector, DenseVector, SparseVector }
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.util.{ Loader, Saveable }
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.HashPartitioner
import org.apache.spark.SparkContext
import org.apache.spark.SparkException
import org.apache.spark.sql.{ Row, SparkSession }
import org.apache.spark.Partitioner
import org.apache.spark.internal.Logging
import scala.collection.mutable.HashMap

/**
 * Information Theoretic Selector model.
 *
 * @param selectedFeatures list of indices to select (filter). Must be ordered asc
 */
@Since("1.6.0")
class InfoThSelectorModel @Since("1.6.0") (
    @Since("1.6.0") val selectedFeatures: Array[Int],
    @Since("1.6.0") val redMap: HashMap[Int, Array[(Int, Float)]] = HashMap[Int, Array[(Int, Float)]]()
) extends VectorTransformer with Saveable {

  require(isSorted(selectedFeatures), "Array has to be sorted asc")

  protected def isSorted(array: Array[Int]): Boolean = {
    var i = 1
    val len = array.length
    while (i < len) {
      if (array(i) < array(i - 1)) return false
      i += 1
    }
    true
  }

  /**
   * Applies transformation on a vector.
   *
   * @param vector vector to be transformed.
   * @return transformed vector.
   */
  @Since("1.3.0")
  override def transform(vector: Vector): Vector = {
    compress(vector, selectedFeatures)
  }

  /**
   * Returns a vector with features filtered.
   * Preserves the order of filtered features the same as their indices are stored.
   * Might be moved to Vector as .slice
   * @param features vector
   * @param filterIndices indices of features to filter, must be ordered asc
   */
  private def compress(features: Vector, filterIndices: Array[Int]): Vector = {
    features match {
      case SparseVector(size, indices, values) =>
        val newSize = filterIndices.length
        val newValues = new ArrayBuilder.ofDouble
        val newIndices = new ArrayBuilder.ofInt
        var i = 0
        var j = 0
        var indicesIdx = 0
        var filterIndicesIdx = 0
        while (i < indices.length && j < filterIndices.length) {
          indicesIdx = indices(i)
          filterIndicesIdx = filterIndices(j)
          if (indicesIdx == filterIndicesIdx) {
            newIndices += j
            newValues += values(i)
            j += 1
            i += 1
          } else {
            if (indicesIdx > filterIndicesIdx) {
              j += 1
            } else {
              i += 1
            }
          }
        }
        // TODO: Sparse representation might be ineffective if (newSize ~= newValues.size)
        Vectors.sparse(newSize, newIndices.result(), newValues.result())
      case DenseVector(values) =>
        val values = features.toArray
        Vectors.dense(filterIndices.map(i => values(i)))
      case other =>
        throw new UnsupportedOperationException(
          s"Only sparse and dense vectors are supported but got ${other.getClass}."
        )
    }
  }

  @Since("1.6.0")
  override def save(sc: SparkContext, path: String): Unit = {
    InfoThSelectorModel.SaveLoadV1_0.save(sc, this, path)
  }

  override protected def formatVersion: String = "1.0"
}

object InfoThSelectorModel extends Loader[InfoThSelectorModel] {
  @Since("1.6.0")
  override def load(sc: SparkContext, path: String): InfoThSelectorModel = {
    InfoThSelectorModel.SaveLoadV1_0.load(sc, path)
  }

  private[feature] object SaveLoadV1_0 {

    private val thisFormatVersion = "1.0"

    /** Model data for import/export */
    case class Data(feature: Int)

    private[feature] val thisClassName = "org.apache.spark.mllib.feature.InfoThSelectorModel"

    def save(sc: SparkContext, model: InfoThSelectorModel, path: String): Unit = {
      val spark = SparkSession.builder().sparkContext(sc).getOrCreate()
      import spark.implicits._
      val metadata = compact(render(
        ("class" -> thisClassName) ~ ("version" -> thisFormatVersion)
      ))
      sc.parallelize(Seq(metadata), 1).saveAsTextFile(Loader.metadataPath(path))

      // Create Parquet data.
      val dataArray = Array.tabulate(model.selectedFeatures.length) { i =>
        Data(model.selectedFeatures(i))
      }
      sc.parallelize(dataArray, 1).toDF().write.parquet(Loader.dataPath(path))

    }

    def load(sc: SparkContext, path: String): InfoThSelectorModel = {
      implicit val formats = DefaultFormats
      val spark = SparkSession.builder().sparkContext(sc).getOrCreate()
      val (className, formatVersion, metadata) = Loader.loadMetadata(sc, path)
      assert(className == thisClassName)
      assert(formatVersion == thisFormatVersion)

      val dataFrame = spark.read.parquet(Loader.dataPath(path))
      val dataArray = dataFrame.select("feature")

      // Check schema explicitly since erasure makes it hard to use match-case for checking.
      Loader.checkSchema[Data](dataFrame.schema)

      val features = dataArray.rdd.map {
        case Row(feature: Int) => (feature)
      }.collect()

      return new InfoThSelectorModel(features)
    }
  }
}

/**
 * Train a feature selection model according to a given criterion
 * and return a subset of data.
 *
 * @param   criterionFactory Initialized criterion to use in this selector
 * @param   nToSelect Maximum number of features to select
 * @param   numPartitions Number of partitions to structure the data.
 * @return  A feature selection model which contains a subset of selected features.
 *
 * Note: LabeledPoint data must be integer values in double representation
 * with a maximum of 256 distinct values. By doing so, data can be transformed
 * to byte class directly, making the selection process much more efficient.
 *
 * Note: numPartitions must be less or equal to the number of features to achieve
 * a better performance. Therefore, the number of histograms to be shuffled is reduced.
 *
 */
@Since("1.6.0")
class InfoThSelector @Since("1.6.0") (
  val criterionFactory: FT,
  val nToSelect: Int = 25,
  val numPartitions: Int = 0
)
    extends Serializable with Logging {

  // Case class for criteria/feature
  protected case class F(feat: Int, crit: Double)
  // Case class for columnar data (dense and sparse version)
  private case class ColumnarData(
    dense: RDD[(Int, Array[Byte])],
    sparse: RDD[(Int, BV[Byte])],
    isDense: Boolean,
    originalNPart: Int
  )
  
  var redundancyMap = new HashMap[Int, Array[(Int, Float)]]()

  /**
   * Performs a info-theory FS process.
   *
   * @param data Columnar data (last element is the class attribute).
   * @param nInstances Number of samples.
   * @param nFeatures Number of features.
   * @return A list with the most relevant features and its scores.
   *
   */
  private[feature] def selectFeatures(
    data: ColumnarData,
    nInstances: Long,
    nFeatures: Int
  ) = {

    val label = nFeatures - 1
    // Initialize all criteria with the relevance computed in this phase. 
    // It also computes and saved some information to be re-used.
    val (it, relevances) = if (data.isDense) {
      val it = InfoTheory.initializeDense(data.dense, label, nInstances, nFeatures, data.originalNPart)
      (it, it.relevances)
    } else {
      val it = InfoTheory.initializeSparse(data.sparse, label, nInstances, nFeatures)
      (it, it.relevances)
    }

    // Initialize all (except the class) criteria with the relevance values
    val pool = Array.fill[InfoThCriterion](nFeatures - 1) {
      val crit = criterionFactory.getCriterion.init(Float.NegativeInfinity)
      crit.setValid(false)
    }
    relevances.collect().foreach {
      case (x, mi) =>
        pool(x) = criterionFactory.getCriterion.init(mi.toFloat)
    }

    // Print most relevant features
    val topByRelevance = relevances.sortBy(_._2, false).take(nToSelect)
    val strRels = topByRelevance.map({ case (f, mi) => (f + 1) + "\t" + "%.4f" format mi })
      .mkString("\n")
    println("\n*** MaxRel features ***\nFeature\tScore\n" + strRels)

    // Get the maximum and initialize the set of selected features with it
    val (max, mid) = pool.zipWithIndex.maxBy(_._1.relevance)
    var selected = Seq(F(mid, max.score))
    pool(mid).setValid(false)

    // MIM does not use redundancy, so for this criterion all the features are selected now
    if (criterionFactory.getCriterion.toString == "MIM") {
      selected = topByRelevance.map({ case (id, relv) => F(id, relv) }).reverse
    }

    var moreFeat = true
    // Iterative process for redundancy and conditional redundancy
    while (selected.size < nToSelect && moreFeat) {

      val redundancies = it match {
        case dit: InfoTheoryDense => dit.getRedundancies(selected.head.feat)
        case sit: InfoTheorySparse => sit.getRedundancies(selected.head.feat)
      }

      println("Target feature: " + selected.head.feat)
      val topk = redundancies.collect().sortBy(_._2._1)
        .slice(0, nToSelect).map{ case(feat, (mi, _)) => feat -> mi}
      redundancyMap += selected.head.feat -> topk
      println("Top Mutual Redundancy: " + topk.mkString("\n"))

      // Update criteria with the new redundancy values      
      redundancies.collect().par.foreach({
        case (k, (mi, cmi)) =>
                   
          pool(k).update(mi.toFloat, cmi.toFloat)
      })

      // select the best feature and remove from the whole set of features
      val (max, maxi) = pool.par.zipWithIndex.filter(_._1.valid).maxBy(_._1)
      if (maxi != -1) {
        selected = F(maxi, max.score) +: selected
        pool(maxi).setValid(false)
      } else {
        moreFeat = false
      }
    }
    selected.reverse
  }

  /**
   * Process in charge of transforming data in a columnar format and launching the FS process.
   *
   * @param data RDD of LabeledPoint.
   * @return A feature selection model which contains a subset of selected features.
   *
   */
  def fit(data: RDD[LabeledPoint]): InfoThSelectorModel = {

    if (data.getStorageLevel == StorageLevel.NONE) {
      logWarning("The input data is not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.")
    }

    // Feature vector must be composed of bytes, not the class
    val requireByteValues = (v: Vector) => {
      val values = v match {
        case sv: SparseVector =>
          sv.values
        case dv: DenseVector =>
          dv.values
      }
      val condition = (value: Double) => value <= 255 && value >= 0.0 && value % 1 == 0.0
      if (!values.forall(condition(_))) {
        val str = values.mkString(",")
        throw new SparkException(
          s"Info-Theoretic Framework requires positive values in range [0, 255] in $str"
        )
      }
    }

    // Get basic info
    val first = data.first
    val dense = first.features.isInstanceOf[DenseVector]
    val nInstances = data.count()
    val nFeatures = first.features.size + 1
    require(nToSelect < nFeatures)

    // Start the transformation to the columnar format
    val colData = if (dense) {

      val originalNPart = data.partitions.size
      val classMap = data.map(_.label).distinct.collect()
        .zipWithIndex.map(t => t._1 -> t._2.toByte)
        .toMap

      // Transform data into a columnar format by transposing the local matrix in each partition
      val eqDistributedData = data.zipWithIndex().map(_.swap).partitionBy(new ExactPartitioner(originalNPart, nInstances))

      val columnarData = eqDistributedData.mapPartitionsWithIndex({ (index, it) =>
        val data = it.toArray.map(_._2)
        val mat = Array.ofDim[Byte](nFeatures, data.length)
        var j = 0
        for (reg <- data) {
          requireByteValues(reg.features)
          for (i <- 0 until reg.features.size) mat(i)(j) = reg.features(i).toByte
          mat(reg.features.size)(j) = classMap(reg.label)
          j += 1
        }

        val chunks = for (i <- 0 until nFeatures) yield ((i * originalNPart + index) -> mat(i))
        chunks.toIterator
      })

      // Sort to group all chunks for the same feature closely. 
      // It will avoid to shuffle too much histograms
      val np = if (numPartitions == 0) nFeatures else numPartitions
      if (np > nFeatures) {
        logWarning("Number of partitions should be equal or less than the number of features."
          + " At least, less than 2x the number of features.")
      }
      val denseData = columnarData.sortByKey(numPartitions = np).persist(StorageLevel.MEMORY_ONLY)
      ColumnarData(denseData, null, true, originalNPart)
    } else {

      val np = if (numPartitions == 0) data.conf.getInt("spark.default.parallelism", 750) else numPartitions
      val classMap = data.map(_.label).distinct.collect()
        .zipWithIndex.map(t => t._1 -> t._2.toByte)
        .toMap

      val sparseData = data.zipWithIndex().flatMap({
        case (lp, r) =>
          requireByteValues(lp.features)
          val sv = lp.features.asInstanceOf[SparseVector]
          val output = (nFeatures - 1) -> (r, classMap(lp.label))
          val inputs = for (i <- 0 until sv.indices.length)
            yield (sv.indices(i), (r, sv.values(i).toByte))
          output +: inputs
      })

      // Transform sparse data into a columnar format 
      // by grouping all values for the same feature in a single vector
      val columnarData = sparseData.groupByKey(new HashPartitioner(np))
        .mapValues({ a =>
          if (a.size >= nInstances) {
            val init = Array.fill[Byte](nInstances.toInt)(0)
            val result: BV[Byte] = new BDV(init)
            a.foreach({ case (k, v) => result(k.toInt) = v })
            result
          } else {
            val init = a.toArray.sortBy(_._1)
            new BSV(init.map(_._1.toInt), init.map(_._2), nInstances.toInt)
          }
        }).persist(StorageLevel.MEMORY_ONLY)

      ColumnarData(null, columnarData, false, data.partitions.size)
    }

    // Start the main algorithm
    val selected = selectFeatures(colData, nInstances, nFeatures)
    if (dense) colData.dense.unpersist() else colData.sparse.unpersist()

    // Print best features according to the mRMR measure
    val out = selected.map {
      case F(feat, rel) =>
        (feat + 1) + "\t" + "%.4f".format(rel)
    }.mkString("\n")
    println("\n*** Selected features ***\nFeature\tScore\n" + out)
    // Features must be sorted
    new InfoThSelectorModel(selected.map { case F(feat, rel) => feat }.sorted.toArray, redundancyMap)
  }
}

class ExactPartitioner(
  partitions: Int,
  elements: Long
)
    extends Partitioner {

  override def numPartitions: Int = partitions

  override def getPartition(key: Any): Int = {
    val k = key.asInstanceOf[Long]
    return (k * partitions / elements).toInt
  }
}
