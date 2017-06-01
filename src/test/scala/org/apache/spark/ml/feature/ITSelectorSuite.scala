package org.apache.spark.ml.feature

import org.apache.spark.sql.{DataFrame, SQLContext}
import org.junit.runner.RunWith
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.scalatest.junit.JUnitRunner
import TestHelper._


/**
  * Test infomartion theoretic feature selection
  *
  * @author Sergio Ramirez
  */
@RunWith(classOf[JUnitRunner])
class ITSelectorSuite extends FunSuite with BeforeAndAfterAll {

  var sqlContext: SQLContext = null

  override def beforeAll(): Unit = {
    sqlContext = new SQLContext(SPARK_CTX)
  }

  /** Do entropy based binning of cars data from UC Irvine repository. */
  test("Run ITFS on colon data (nPart = 10, nfeat = 10)") {

    val df = readColonData(sqlContext)
    val cols = df.columns
    val pad = 2
    val allVectorsDense = true
    val model = getSelectorModel(sqlContext, df, cols.drop(1), cols.head, 10, 10, allVectorsDense, pad)

    assertResult("512, 764, 1324, 1380, 1411, 1422, 1581, 1670, 1671, 1971") {
      model.selectedFeatures.mkString(", ")
    }
  }

  

}