package org.apache.spark.ml.feature

import org.apache.spark.sql.{DataFrame, SQLContext}
import org.junit.runner.RunWith
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.scalatest.junit.JUnitRunner
import TestHelper._


/**
  * Test information theoretic feature selection on datasets from Peng's webpage
  *
  * @author Sergio Ramirez
  */
@RunWith(classOf[JUnitRunner])
class ITSelectorSuite extends FunSuite with BeforeAndAfterAll {

  var sqlContext: SQLContext = null

  override def beforeAll(): Unit = {
    sqlContext = new SQLContext(SPARK_CTX)
  }

  /** Do mRMR feature selection on COLON data. */
  test("Run ITFS on colon data (nPart = 10, nfeat = 10)") {

    val df = readCSVData(sqlContext, "test_colon_s3.csv")
    val cols = df.columns
    val pad = 2
    val allVectorsDense = true
    val model = getSelectorModel(sqlContext, df, cols.drop(1), cols.head, 
        10, 10, allVectorsDense, pad)

    assertResult("512, 764, 1324, 1380, 1411, 1422, 1581, 1670, 1671, 1971") {
      model.selectedFeatures.mkString(", ")
    }
  }

  /** Do mRMR feature selection on LEUKEMIA data. */
  test("Run ITFS on leukemia data (nPart = 10, nfeat = 10)") {

    val df = readCSVData(sqlContext, "test_leukemia_s3.csv")
    val cols = df.columns
    val pad = 2
    val allVectorsDense = true
    val model = getSelectorModel(sqlContext, df, cols.drop(1), cols.head, 
        10, 10, allVectorsDense, pad)

    assertResult("1084, 1719, 1774, 1822, 2061, 2294, 3192, 4387, 4787, 6795") {
      model.selectedFeatures.mkString(", ")
    }
  }
  
  /** Do mRMR feature selection on LUNG data. */
  test("Run ITFS on lung data (nPart = 10, nfeat = 10)") {

    val df = readCSVData(sqlContext, "test_lung_s3.csv")
    val cols = df.columns
    val pad = 2
    val allVectorsDense = true
    val model = getSelectorModel(sqlContext, df, cols.drop(1), cols.head, 
        10, 10, allVectorsDense, pad)

    assertResult("18, 22, 29, 125, 132, 150, 166, 242, 243, 269") {
      model.selectedFeatures.mkString(", ")
    }
  }

  /** Do mRMR feature selection on LYMPHOMA data. */
  test("Run ITFS on lymphoma data (nPart = 10, nfeat = 10)") {

    val df = readCSVData(sqlContext, "test_lymphoma_s3.csv")
    val cols = df.columns
    val pad = 2
    val allVectorsDense = true
    val model = getSelectorModel(sqlContext, df, cols.drop(1), cols.head, 
        10, 10, allVectorsDense, pad)

    assertResult("236, 393, 759, 2747, 2818, 2841, 2862, 3014, 3702, 3792") {
      model.selectedFeatures.mkString(", ")
    }
  }

  /** Do mRMR feature selection on NCI data. */
  test("Run ITFS on nci data (nPart = 10, nfeat = 10)") {

    val df = readCSVData(sqlContext, "test_nci9_s3.csv")
    val cols = df.columns
    val pad = 2
    val allVectorsDense = true
    val model = getSelectorModel(sqlContext, df, cols.drop(1), cols.head, 
        10, 10, allVectorsDense, pad)

    assertResult("443, 755, 1369, 1699, 3483, 5641, 6290, 7674, 9399, 9576") {
      model.selectedFeatures.mkString(", ")
    }
  }
}