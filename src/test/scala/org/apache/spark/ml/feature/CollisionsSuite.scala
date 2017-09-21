package org.apache.spark.ml.feature

import org.apache.spark.sql.{DataFrame, SQLContext}
import org.junit.runner.RunWith
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.scalatest.junit.JUnitRunner
import TestHelper._
import org.apache.spark.sql.Row
import org.apache.spark.ml.linalg._
import scala.collection.mutable.HashSet
import scala.collection.mutable.TreeSet
import com.github.karlhigley.spark.neighbors.KNN
import com.github.karlhigley.spark.neighbors.ANNModel.IDPoint
import com.github.karlhigley.spark.neighbors.util._

import org.apache.spark.util.LongAccumulator


/**
  * Test information theoretic feature selection on datasets from Peng's webpage
  *
  * @author Sergio Ramirez
  */
@RunWith(classOf[JUnitRunner])
class CollisionsSuite extends FunSuite with BeforeAndAfterAll {

  var sqlContext: SQLContext = null
  private val log2 = { x: Float => math.log(x) / math.log(2) }

  override def beforeAll(): Unit = {
    sqlContext = new SQLContext(SPARK_CTX)
  }
  
  
  def initRDD(df: DataFrame, allVectorsDense: Boolean, padded: Int) = {
    
    val featureAssembler = new VectorAssembler()
      .setInputCols(df.columns.drop(1))
      .setOutputCol("features")
    val processedDf0 = cleanLabelCol(df, df.columns.head)
    val processedDf = featureAssembler.transform(processedDf0)
      .select(processedDf0.columns.head + INDEX_SUFFIX, "features")
    val rdd = processedDf.rdd.map {
      case Row(label: Double, features: Vector) =>
        val standardv = if(allVectorsDense){
          Vectors.dense(features.toArray.map(_ + padded))
        } else {
            val sparseVec = features.toSparse
            val newValues: Array[Double] = sparseVec.values.map(_ + padded)
            Vectors.sparse(sparseVec.size, sparseVec.indices, newValues)
        }        
        LabeledPoint(label, standardv)
    }
    rdd
  }
  
  def compareSolutions(df: DataFrame, allVectorsDense: Boolean, padded: Int, 
      redundancyMatrix: breeze.linalg.DenseMatrix[Float]) {
    
    // Return results
    val numTopFeatures = 10
    val nTop = 10
    val model = getSelectorModel(sqlContext, df, df.columns.drop(1), df.columns.head, 
        10, numTopFeatures = numTopFeatures, allVectorsDense, padded)
    val nf = redundancyMatrix.cols
    val order = 1 // Note: -1 means descending
    
    model.selectedFeatures.foreach{sf => 
        model.redMap.get(sf) match {
          case Some(a) => 
            val redInfo = a.sortBy(_._2 * order).slice(0, nTop).map(_._1).toSet
            val redCollisions = redundancyMatrix(::,sf).toArray.zipWithIndex
              .filter(n => n._2 != nf - 1 && n._2 != sf)
              .sortBy(_._1 * order)
              .slice(0, nTop)
           println("Feature target: " + sf)
           println("# distinct features: " + redInfo.diff(redCollisions.map(_._2).toSet).toString())
           println("Values: " + redCollisions.mkString("\n"))
          case None => println("That didn't work.")
        }
    }
    
  }
  
  test("Run collision estimation on lung data (nPart = 10, nfeat = 10)") {

    
    val df = readCSVData(sqlContext, "test_lung_s3.csv")
    val padded = 2
    val allVectorsDense = true
    val rdd = initRDD(df, allVectorsDense, padded).zipWithUniqueId()
    
    val elements = rdd.collect
    val nf = elements.head._1.features.size + 1
    val belems = rdd.context.broadcast(elements)
    
    val accMarginal = new VectorAccumulator(nf)
    // Then, register it into spark context:
    rdd.context.register(accMarginal, "marginal")
    val accJoint = new MatrixAccumulator(nf, nf)
    rdd.context.register(accJoint, "joint")
    val accConditional = new MatrixAccumulator(nf, nf)
    rdd.context.register(accConditional, "conditional")
    val total = rdd.context.longAccumulator("total")
    println("# instances: " + rdd.count())
    rdd.foreachPartition { it =>
        
        val marginal = breeze.linalg.DenseVector.zeros[Long](nf)
        val last = marginal.size - 1
        val joint = breeze.linalg.DenseMatrix.zeros[Long](nf, nf)
        val condjoint = breeze.linalg.DenseMatrix.zeros[Long](nf, nf)
        val others = belems.value
      
        while(it.hasNext){
          val (e1, id1) = it.next
          
          others.foreach{ case (e2, id2) => 
            
            if(id1 > id2) {
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
    compareSolutions(df, allVectorsDense, padded, redundancyMatrix)

  } 
  
  /** Do mRMR feature selection on lung data. */
  /*test("Run collision estimation on lung data (nPart = 10, nfeat = 10)") {

    val df = readCSVData(sqlContext, "test_lung_s3.csv")
    val padded = 2
    val allVectorsDense = true
    
    val featureAssembler = new VectorAssembler()
      .setInputCols(df.columns.drop(1))
      .setOutputCol("features")
    val processedDf0 = cleanLabelCol(df, df.columns.head)
    val processedDf = featureAssembler.transform(processedDf0)
      .select(processedDf0.columns.head + INDEX_SUFFIX, "features")
    val rdd = processedDf.rdd.map {
      case Row(label: Double, features: Vector) =>
        val standardv = if(allVectorsDense){
          Vectors.dense(features.toArray.map(_ + padded))
        } else {
            val sparseVec = features.toSparse
            val newValues: Array[Double] = sparseVec.values.map(_ + padded)
            Vectors.sparse(sparseVec.size, sparseVec.indices, newValues)
        }        
        LabeledPoint(label, standardv)
    }
    
    val elements = rdd.collect
    val nf = elements.head.features.size + 1
    val marginal = breeze.linalg.DenseVector.zeros[Float](nf)
    val last = marginal.size - 1
    val joint = breeze.linalg.DenseMatrix.zeros[Float](nf, nf)
    val condjoint = breeze.linalg.DenseMatrix.zeros[Float](nf, nf)
    var total = 0.0f
    (0 until elements.size).map{ i =>
      val top = i + 1
      (top until elements.size).map{ j =>
        var set = new TreeSet[Int]()
        val clshit = elements(i).label == elements(j).label
        elements(i).features.foreachActive{ (index, value) =>
           if(elements(i).features(index) == elements(j).features(index)){
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
        total += 1
      }
    }
    
    marginal :/= total
    joint :/= total
    condjoint :/= total
    
    // Compute mutual information using collisions with and without class
    val redundancyMatrix = breeze.linalg.DenseMatrix.zeros[Float](nf, nf)
    joint.activeIterator.foreach{ case((i1,i2), value) =>
      if(i1 < i2 && i1 != last && i2 != last){
        val red = value * log2(value / (marginal(i1) * marginal(i2))).toFloat
        redundancyMatrix(i1, i2) = red
        redundancyMatrix(i2, i1) = red        
      }
        
    }
    val asd0 = redundancyMatrix(::,22).toArray.zipWithIndex.sortBy(_._1).slice(0, 15)
    println("Joint: " + asd0.mkString("\n"))
    
    val condRedunMatrix = breeze.linalg.DenseMatrix.zeros[Float](nf, nf)
    condjoint.activeIterator.foreach{ case((i1,i2), value) =>
      if(i1 < i2){
        val red = value * log2((marginal(last) *  value) / (joint(i1, last) * joint(i2, last))).toFloat
        condRedunMatrix(i1, i2) = red
        condRedunMatrix(i2, i1) = red 
      }       
    }
    val asd = condRedunMatrix(::,22).toArray.zipWithIndex.sortBy(_._1).slice(0, 15)
    println("Conditional: " + asd.mkString("\n"))
  } */
    
    /** Do mRMR feature selection on lung data. */
  /*test("Run collision estimation on lung data using KNN (nPart = 10, nfeat = 10)") {

    val df = readCSVData(sqlContext, "test_lung_s3.csv")
    val padded = 2
    val allVectorsDense = true
    
    val featureAssembler = new VectorAssembler()
      .setInputCols(df.columns.drop(1))
      .setOutputCol("features")
    val processedDf0 = cleanLabelCol(df, df.columns.head)
    val processedDf = featureAssembler.transform(processedDf0).select(processedDf0.columns.head + INDEX_SUFFIX, "features")
    val rdd = processedDf.rdd.map {
      case Row(label: Double, features: Vector) =>
        val standardv = if(allVectorsDense){
          Vectors.dense(features.toArray.map(_ + padded))
        } else {
            val sparseVec = features.toSparse
            val newValues: Array[Double] = sparseVec.values.map(_ + padded)
            Vectors.sparse(sparseVec.size, sparseVec.indices, newValues)
        }        
        LabeledPoint(label, standardv)
    }
    
    // KNN wrapper
    val model = new KNN().train(rdd)
    val queryPoints = rdd.sample(withReplacement = false, fraction = 1.0)
    val neighbors = model.neighbors(queryPoints, 10).collect()
    
    val nf = neighbors.head._1.features.size + 1
    val marginal = breeze.linalg.DenseVector.zeros[Float](nf)
    val last = marginal.size - 1
    val joint = breeze.linalg.DenseMatrix.zeros[Float](nf, nf)
    val condjoint = breeze.linalg.DenseMatrix.zeros[Float](nf, nf)
    var total = 0.0f
    neighbors.foreach{ case(e1, neigs) =>
      neigs.foreach{ e2 =>
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
        total += 1
      }
    }
    
    marginal :/= total
    joint :/= total
    condjoint :/= total
    
    // Compute mutual information using collisions with and without class
    val redundancyMatrix = breeze.linalg.DenseMatrix.zeros[Float](nf, nf)
    joint.activeIterator.foreach{ case((i1,i2), value) =>
      if(i1 < i2 && i1 != last && i2 != last){
        val red = value * log2(value / (marginal(i1) * marginal(i2))).toFloat
        redundancyMatrix(i1, i2) = red
        redundancyMatrix(i2, i1) = red        
      }
        
    }
    val asd0 = redundancyMatrix(::,22).toArray.zipWithIndex.sortBy(-_._1).slice(0, 15)
    println("Joint: " + asd0.mkString("\n"))
    
    val condRedunMatrix = breeze.linalg.DenseMatrix.zeros[Float](nf, nf)
    condjoint.activeIterator.foreach{ case((i1,i2), value) =>
      if(i1 < i2  && i1 != last && i2 != last){
        val red = value * log2((marginal(last) *  value) / (joint(i1, last) * joint(i2, last))).toFloat
        condRedunMatrix(i1, i2) = red
        condRedunMatrix(i2, i1) = red 
      }       
    }
    val asd = condRedunMatrix(::,22).toArray.zipWithIndex.sortBy(-_._1).slice(0, 15)
    println("Conditional: " + asd.mkString("\n"))
  }*/
  
  
  
    /** Do mRMR feature selection on LUNG data. */
  /*test("Run ITFS on lung data (nPart = 10, nfeat = 10)") {

    val df = readCSVData(sqlContext, "test_lung_s3.csv")
    val cols = df.columns
    val pad = 2
    val allVectorsDense = true
    val model = getSelectorModel(sqlContext, df, cols.drop(1), cols.head, 
        10, 10, allVectorsDense, pad)

    assertResult("18, 22, 29, 125, 132, 150, 166, 242, 243, 269") {
      model.selectedFeatures.mkString(", ")
    }
  }*/
  
}