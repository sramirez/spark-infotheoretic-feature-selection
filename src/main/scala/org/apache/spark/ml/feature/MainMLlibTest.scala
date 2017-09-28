package org.apache.spark.ml.feature

import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.sql.Row
import scala.collection.mutable.TreeSet
import org.apache.spark.util.LongAccumulator
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StructType
import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.SparkConf
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.ml.Pipeline


/**
 * @author sramirez
 */


object MainMLlibTest {
  var sqlContext: SQLContext = null
  var pathFile = "test_lung_s3.csv"
  var order = 1 // Note: -1 means descending
  var nPartitions = 1
  var nTop = 10
  var discretize = false
  var padded = 2
  var classLastIndex = false
  var clsLabel: String = null
  var inputLabel: String = "features"
  var firstHeader: Boolean = false

  def main(args: Array[String]) {
    
    val initStartTime = System.nanoTime()
    
    val conf = new SparkConf().setAppName("CollisionFS Test").setMaster("local[*]")
    val sc = new SparkContext(conf)
    sqlContext = new SQLContext(sc)

    println("Usage: MLlibTest --train-file=\"hdfs://blabla\" --npart=1 --ntop=10 --disc=false --padded=2 --class-last=true --header=false")
        
    // Create a table of parameters (parsing)
    val params = args.map{ arg =>
        val param = arg.split("--|=").filter(_.size > 0)
        param.size match {
          case 2 =>  param(0) -> param(1)
          case _ =>  "" -> ""
        }
    }.toMap    
    
    pathFile = params.getOrElse("train-file", "src/test/resources/data/test_lung_s3.csv")
    nPartitions = params.getOrElse("npart", "1").toInt
    nTop = params.getOrElse("ntop", "10").toInt
    discretize = params.getOrElse("disc", "false").toBoolean
    padded = params.getOrElse("padded", "2").toInt
    classLastIndex = params.getOrElse("class-last", "false").toBoolean
    firstHeader = params.getOrElse("header", "false").toBoolean
    
    println("Params used: " +  params.mkString("\n"))
    
    doComparison()
  }
  
  def doComparison() {
    val rawDF = TestHelper.readCSVData(sqlContext, pathFile, firstHeader)
    val df = preProcess(rawDF).select(clsLabel, inputLabel)
    val allVectorsDense = true
    
    println("df: " + df.first().toString())
    
    val origRDD = initRDD(df, allVectorsDense)
    val rdd = origRDD.map {
      case Row(label: Double, features: Vector) =>
        LabeledPoint(label, features)
    }.repartition(nPartitions).cache //zipwithUniqueIndexs
    
    println("rdd: " + rdd.first().toString())
    
    //val elements = rdd.collect
    val nf = rdd.first.features.size + 1
    //val belems = rdd.context.broadcast(elements)
    
    val accMarginal = new VectorAccumulator(nf)
    // Then, register it into spark context:
    rdd.context.register(accMarginal, "marginal")
    val accJoint = new MatrixAccumulator(nf, nf)
    rdd.context.register(accJoint, "joint")
    val accConditional = new MatrixAccumulator(nf, nf)
    rdd.context.register(accConditional, "conditional")
    val total = rdd.context.longAccumulator("total")
    println("# instances: " + rdd.count)
    println("# partitions: " + rdd.partitions.size)
    
    rdd.foreachPartition { it =>
        
        val marginal = breeze.linalg.DenseVector.zeros[Long](nf)
        val last = marginal.size - 1
        val joint = breeze.linalg.DenseMatrix.zeros[Long](nf, nf)
        val condjoint = breeze.linalg.DenseMatrix.zeros[Long](nf, nf)
        //val others = belems.value
        val elements = it.toArray
        (0 until elements.size).map{ id1 =>
        //while(it.hasNext){
          //val (e1, id1) = it.next
          val e1 = elements(id1)
          //others.foreach{ case (e2, id2) => 
          (id1 + 1 until elements.size).map{ id2 => 
            val e2 = elements(id2)
            //if(id1 > id2) {
            if(true) {
                var set = new TreeSet[Int]()
                val clshit = e1.label == e2.label
                e1.features.foreachActive{ (index, value) =>
                   if(e1.features(index) == e2.features(index)){
                       marginal(index) += 1
                       set += index
                   }            
                }
                
                // Count matches in output feature
                if(clshit){          
                  marginal(last) += 1
                  set += last
                }
                
                // Generate combinations and update joint collisions counter
                val arr = set.toArray
                (0 until arr.size).map{f1 => 
                  (f1 + 1 until arr.size).map{ f2 =>
                    joint(arr(f1), arr(f2)) += 1
                    if(clshit)
                      condjoint(arr(f1), arr(f2)) += 1
                  }         
                }
              total.add(1L)
            }            
        }
      }
        
      accMarginal.add(marginal)
      accJoint.add(joint)
      accConditional.add(condjoint)
    }
  
    val marginal = accMarginal.value.mapValues(_.toFloat) 
    marginal :/= total.value.toFloat 
    val joint = accJoint.value.toDenseMatrix.mapValues(_.toFloat)
    joint :/= total.value.toFloat 
    val conditional = accConditional.value.toDenseMatrix.mapValues(_.toFloat)
    conditional :/= total.value.toFloat
  
    // Compute mutual information using collisions with and without class
    val redundancyMatrix = breeze.linalg.DenseMatrix.zeros[Float](nf, nf)
    joint.activeIterator.foreach { case((i1,i2), value) =>
      if(i1 < i2 && i1 != nf - 1 && i2 != nf - 1){
        val red = value * log2(value / (marginal(i1) * marginal(i2))).toFloat        
        
        redundancyMatrix(i1, i2) = red
        redundancyMatrix(i2, i1) = red        
      }
        
    }
    compareSolutions(origRDD, df.schema, redundancyMatrix)
  }
  
  
  def preProcess(df: DataFrame) = {
    val other = if(classLastIndex) df.columns.dropRight(1) else df.columns.drop(1)
    clsLabel = if(classLastIndex) df.columns.last else df.columns.head
    
    // Index categorical values
    val stringTypes = df.dtypes.filter(_._2 == "StringType").map(_._1)
    val tmpNames = df.dtypes.map{ case(name, typ) => if(typ == "StringType") name + "-indexed" else name}
    clsLabel = if(classLastIndex) tmpNames.last else tmpNames.head
    val newNames = if(classLastIndex) tmpNames.dropRight(1) else tmpNames.drop(1)
    val indexers = stringTypes.map{ name =>
        new StringIndexer()
          .setInputCol(name)
          .setOutputCol(name + "-indexed")
    }

    val pipeline = new Pipeline().setStages(indexers)
    val typedDF = pipeline.fit(df).transform(df).drop(stringTypes: _*)

    // Clean Label Column
    val cleanedDF = TestHelper.cleanLabelCol(typedDF, clsLabel)
    clsLabel = clsLabel + TestHelper.INDEX_SUFFIX
    println("clslabel: " + clsLabel)
    
    // Assemble all input features
    val featureAssembler = new VectorAssembler()
      .setInputCols(newNames)
      .setOutputCol(inputLabel)

    var processedDF = featureAssembler.transform(cleanedDF)
      .select(clsLabel, inputLabel)

    println("clsLabel: " + clsLabel)
    println("Columns: " + processedDF.columns.mkString(","))
    println("Schema: " + processedDF.schema.toString)
    println(processedDF.first.get(1))
      
    if(discretize){      
      val discretizer = new MDLPDiscretizer()
        .setMaxBins(15)
        .setMaxByPart(10000)
        .setInputCol(inputLabel)
        .setLabelCol(clsLabel)
        .setOutputCol("disc-" + inputLabel)
        
      inputLabel = "disc-" + inputLabel
      
      val model = discretizer.fit(processedDF)
      processedDF = model.transform(processedDF)
      processedDF.show
    }
    processedDF
  }
   
  def initRDD(df: DataFrame, allVectorsDense: Boolean) = {
    val pad = padded // Needed to avoid task not serializable exception (raised by the class by itself)
    df.rdd.map {
      case Row(label: Double, features: Vector) =>
        val standardv = if(allVectorsDense){
          Vectors.dense(features.toArray.map(_ + pad))
        } else {
            val sparseVec = features.toSparse
            val newValues: Array[Double] = sparseVec.values.map(_ + pad)
            Vectors.sparse(sparseVec.size, sparseVec.indices, newValues)
        }        
        Row.fromSeq(Seq(label, standardv))
    }
  }
  
  def compareSolutions(rdd: RDD[Row], schema: StructType,
      redundancyMatrix: breeze.linalg.DenseMatrix[Float]) {
    
    // Return results
    val inputData = sqlContext.createDataFrame(rdd, schema).cache()
    val cls = if(classLastIndex) inputData.columns.last else inputData.columns.head
    println("Columns class: " + cls)
    
    //println("inputData: " + inputData.first.toString())
    println("schema: " + schema.toString())
    
    val selector = new InfoThSelector(redundancyMatrix)
        .setSelectCriterion("mrmr")
        .setNPartitions(nPartitions)
        .setNumTopFeatures(nTop)
        .setFeaturesCol(inputLabel) // this must be a feature vector
        .setLabelCol(clsLabel)
        .setOutputCol("selectedFeatures")
        
    val model = selector.fit(inputData)
    compareRedundancies(model, redundancyMatrix, order)    
  }
  
  def compareRedundancies(model: InfoThSelectorModel, 
      redundancyMatrix: breeze.linalg.DenseMatrix[Float],
      order: Int) {
        
    model.selectedFeatures.foreach{ sf => 
        model.redMap.get(sf) match {
          case Some(redByFeature) =>
            // InfoTheoretic
            val avgInfo = redByFeature.map(_._2._1).sum / redByFeature.length
            val devInfo = Math.sqrt((redByFeature.map(_._2._1).map(_ - avgInfo).map(t => t*t).sum) / redByFeature.length).toFloat
            
            val normRedInfo = redByFeature.map{ case(id, (score, _)) => id -> ((score - avgInfo) / devInfo)}
            val rankingInfoTh = normRedInfo
              .sortBy(_._2 * order)  
              .slice(0, nTop)
              .zipWithIndex
              .map{case ((id, score), rank) =>
                id -> (rank, score)  
              }
              
            
            // Collision version
            val rawRedColl = redundancyMatrix(::,sf).toArray
              .dropRight(1)
              .zipWithIndex
              .filter(_._2 != sf)
            
            val avgColl = rawRedColl.map(_._1).sum / rawRedColl.length
            val devColl = Math.sqrt((rawRedColl.map(_._1).map( _ - avgColl).map(t => t*t).sum) / rawRedColl.length).toFloat
            
            val normRedColl = rawRedColl.map{ case(score, id) => (score - avgColl) / devColl.toFloat -> id} 
            val rankingCollisions = normRedColl
              .sortBy(_._1 * order)
              .zipWithIndex
              .map{case ((score, id), rank) =>
                id -> (rank, score)  
              }
              
             // Compute average distance between rankings
            val mapRankingCollisions = rankingCollisions.toMap
            var sumDifRankings = 0.0f
            rankingInfoTh.foreach{ case(f, (rank1, _)) =>
              val (rank2, _) = mapRankingCollisions.getOrElse(f, -1 -> -1.0f)
              if(rank2 > 0) {
                sumDifRankings += math.abs(rank1 - rank2)
              }
            }
              
            // Compute average distance between scores
            var sumDifScores = 0.0f
            rankingInfoTh.foreach{ case(f, (_, score1)) =>
              val (_, score2) = mapRankingCollisions.getOrElse(f, -1 -> Float.NaN)
              if(score2 != Float.NaN) {
                sumDifScores += math.abs(score1 - score2)
              }
            }
              
            // Compute final statistics
            println("Feature target: " + sf)            
            println("Avg. ranking distance by feature: " + sumDifRankings / rankingInfoTh.size)
            println("Avg. score distance by feature: " + sumDifScores / rankingInfoTh.size)
            println("# distinct features: " + rankingInfoTh.map(_._1).toSet.diff(
                rankingCollisions.slice(0, nTop).map(_._1).toSet))           
            println("InfoTh Values: " + rankingInfoTh.mkString("\n"))
            println("Collision Values: " + rankingCollisions.slice(0, nTop).mkString("\n"))
           
          case None => println("That didn't work.")
        }
    }
  }
  
  def log2(x: Float) = { math.log(x) / math.log(2) }
}
