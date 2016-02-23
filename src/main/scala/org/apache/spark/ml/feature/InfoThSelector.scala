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

package org.apache.spark.ml.feature

import org.apache.hadoop.fs.Path
import org.apache.spark.annotation.{Experimental, Since}
import org.apache.spark.ml._
import org.apache.spark.ml.attribute.{AttributeGroup, _}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.mllib.feature
import org.apache.spark.mllib.linalg.{Vector, VectorUDT}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.mllib.feature.InfoThCriterionFactory

/**
 * Params for [[InfoThSelector]] and [[InfoThSelectorModel]].
 */
private[feature] trait InfoThSelectorParams extends Params
  with HasFeaturesCol with HasOutputCol with HasLabelCol {

  /**
   * Information Theoretic criterion used to rank the features. The default value is the criterion mRMR.
   * @group param
   */
  
  val MIM = "mim"
  val MIFS = "mifs"
  val JMI  = "jmi"
  val MRMR = "mrmr"
  val ICAP = "icap"
  val CMIM = "cmim"
  val IF   = "if"

  final val selectCriterion = new Param[String](this, "selectCriterion",
    "Information Theoretic criterion used to rank the features. The criterion to be chosen are: (mim, mifs, jmi, mrmr, icap, cmim, if).")
  setDefault(selectCriterion -> "mrmr")
  
  /**
   * Number of features that selector will select (ordered by statistic value descending). If the
   * number of features is < numTopFeatures, then this will select all features. The default value
   * of numTopFeatures is 50.
   * @group param
   */
  final val numTopFeatures = new IntParam(this, "numTopFeatures",
    "Number of features that selector will select, ordered by statistics value descending. If the" +
      " number of features is < numTopFeatures, then this will select all features.",
    ParamValidators.gtEq(1))
  setDefault(numTopFeatures -> 25)
  
  /**
   * Number of partitions to use after the data matrix is transformed to a columnar format. The default value
   * is 0, which means that the default level of parallelism is used.
   * @group param
   */
  final val nPartitions = new IntParam(this, "nPartitions",
    "Number of partitions to use after the data matrix is transformed to a columnar format.",
    ParamValidators.gtEq(0))
  setDefault(nPartitions -> 0)
  
  /** @group getParam */
  def getSelectCriterion: String = $(selectCriterion)
  
  /** @group getParam */
  def getNumTopFeatures: Int = $(numTopFeatures)

  /** @group getParam */
  def getNPartitions: Int = $(nPartitions)
}

/**
 * :: Experimental ::
 * Chi-Squared feature selection, which selects categorical features to use for predicting a
 * categorical label.
 */
@Experimental
final class InfoThSelector(override val uid: String)
  extends Estimator[InfoThSelectorModel] with InfoThSelectorParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("InfoThSelector"))

  /** @group setParam */
  def setSelectCriterion(value: String): this.type = set(selectCriterion, value)
  
  /** @group setParam */
  def setNumTopFeatures(value: Int): this.type = set(numTopFeatures, value)
  
  /** @group setParam */
  def setNPartitions(value: Int): this.type = set(nPartitions, value)
  
  /** @group setParam */
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /** @group setParam */
  def setLabelCol(value: String): this.type = set(labelCol, value)

  override def fit(dataset: DataFrame): InfoThSelectorModel = {
    transformSchema(dataset.schema, logging = true)
    val input = dataset.select($(labelCol), $(featuresCol)).map {
      case Row(label: Double, features: Vector) =>
        LabeledPoint(label, features)
    }

    val InfoThSelector = new feature.InfoThSelector(
        new InfoThCriterionFactory($(selectCriterion)),
        $(numTopFeatures),
        $(nPartitions)).fit(input)
    copyValues(new InfoThSelectorModel(uid, InfoThSelector).setParent(this))
  }

  override def transformSchema(schema: StructType): StructType = {
    validateParams()
    SchemaUtils.checkColumnType(schema, $(featuresCol), new VectorUDT)
    SchemaUtils.checkColumnType(schema, $(labelCol), DoubleType)
    SchemaUtils.appendColumn(schema, $(outputCol), new VectorUDT)
  }

  override def copy(extra: ParamMap): InfoThSelector = defaultCopy(extra)
}

@Since("1.6.0")
object InfoThSelector extends DefaultParamsReadable[InfoThSelector] {

  @Since("1.6.0")
  override def load(path: String): InfoThSelector = super.load(path)
}

/**
 * :: Experimental ::
 * Model fitted by [[InfoThSelector]].
 */
@Experimental
final class InfoThSelectorModel private[ml] (
    override val uid: String,
    private val InfoThSelector: feature.InfoThSelectorModel)
  extends Model[InfoThSelectorModel] with InfoThSelectorParams with MLWritable {

  import InfoThSelectorModel._

  /** list of indices to select (filter). Must be ordered asc */
  val selectedFeatures: Array[Int] = InfoThSelector.selectedFeatures

  /** @group setParam */
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /** @group setParam */
  def setLabelCol(value: String): this.type = set(labelCol, value)

  override def transform(dataset: DataFrame): DataFrame = {
    val transformedSchema = transformSchema(dataset.schema, logging = true)
    val newField = transformedSchema.last
    val selector = udf { InfoThSelector.transform _ }
    dataset.withColumn($(outputCol), selector(col($(featuresCol))), newField.metadata)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateParams()
    SchemaUtils.checkColumnType(schema, $(featuresCol), new VectorUDT)
    val newField = prepOutputField(schema)
    val outputFields = schema.fields :+ newField
    StructType(outputFields)
  }

  /**
   * Prepare the output column field, including per-feature metadata.
   */
  private def prepOutputField(schema: StructType): StructField = {
    val selector = InfoThSelector.selectedFeatures.toSet
    val origAttrGroup = AttributeGroup.fromStructField(schema($(featuresCol)))
    val featureAttributes: Array[Attribute] = if (origAttrGroup.attributes.nonEmpty) {
      origAttrGroup.attributes.get.zipWithIndex.filter(x => selector.contains(x._2)).map(_._1)
    } else {
      Array.fill[Attribute](selector.size)(NominalAttribute.defaultAttr)
    }
    val newAttributeGroup = new AttributeGroup($(outputCol), featureAttributes)
    newAttributeGroup.toStructField()
  }

  override def copy(extra: ParamMap): InfoThSelectorModel = {
    val copied = new InfoThSelectorModel(uid, InfoThSelector)
    copyValues(copied, extra).setParent(parent)
  }

  @Since("1.6.0")
  override def write: MLWriter = new InfoThSelectorModelWriter(this)
}

@Since("1.6.0")
object InfoThSelectorModel extends MLReadable[InfoThSelectorModel] {

  private[InfoThSelectorModel]
  class InfoThSelectorModelWriter(instance: InfoThSelectorModel) extends MLWriter {

    private case class Data(selectedFeatures: Seq[Int])

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val data = Data(instance.selectedFeatures.toSeq)
      val dataPath = new Path(path, "data").toString
      sqlContext.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  private class InfoThSelectorModelReader extends MLReader[InfoThSelectorModel] {

    private val className = classOf[InfoThSelectorModel].getName

    override def load(path: String): InfoThSelectorModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val dataPath = new Path(path, "data").toString
      val data = sqlContext.read.parquet(dataPath).select("selectedFeatures").head()
      val selectedFeatures = data.getAs[Seq[Int]](0).toArray
      val oldModel = new feature.InfoThSelectorModel(selectedFeatures)
      val model = new InfoThSelectorModel(metadata.uid, oldModel)
      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }

  @Since("1.6.0")
  override def read: MLReader[InfoThSelectorModel] = new InfoThSelectorModelReader

  @Since("1.6.0")
  override def load(path: String): InfoThSelectorModel = super.load(path)
}
