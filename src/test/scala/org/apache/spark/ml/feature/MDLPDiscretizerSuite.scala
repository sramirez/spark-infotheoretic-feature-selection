package org.apache.spark.ml.feature

import org.apache.spark.sql.{DataFrame, SQLContext}
import org.junit.runner.RunWith
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.scalatest.junit.JUnitRunner
import TestHelper._


/**
  * Test MDLP discretization
  *
  * @author Barry Becker
  */
@RunWith(classOf[JUnitRunner])
class MDLPDiscretizerSuite extends FunSuite with BeforeAndAfterAll {

  var sqlContext: SQLContext = null

  override def beforeAll(): Unit = {
    sqlContext = new SQLContext(SPARK_CTX)
  }

  /** Do entropy based binning of cars data from UC Irvine repository. */
  test("Run MDLPD on single mpg column in cars data (maxBins = 10)") {

    val df = readColonData(sqlContext)
    val model = getSelectorModel(df, Array("mpg"), "origin", 10)

    assertResult("764, 1581, 1671, 512, 1670, 1324, 1381, 1971, 1422, 1411") {
      model.selectedFeatures.mkString(", ")
    }
  }

  

}