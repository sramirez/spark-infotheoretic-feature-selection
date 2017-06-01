package org.apache.spark.ml.feature

import java.sql.Timestamp
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.functions._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.types._
import org.joda.time.format.DateTimeFormat
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.VectorUDT
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.util._

/**
  * Loads various test datasets
  */
object TestHelper {

  final val SPARK_CTX = createSparkContext()
  final val FILE_PREFIX = "src/test/resources/data/"
  final val ISO_DATE_FORMAT = DateTimeFormat.forPattern("yyyy-MM-dd'T'HH:mm:ss")
  final val NULL_VALUE = "?"

  // This value is used to represent nulls in string columns
  final val MISSING = "__MISSING_VALUE__"
  final val CLEAN_SUFFIX: String = "_CLEAN"
  final val INDEX_SUFFIX: String = "_IDX"

  /**
    * @return the feature select fit to the data given the specified features to bin and label use as target.
    */
  
  def createSelectorModel(sqlContext: SQLContext, dataframe: Dataset[_], inputCols: Array[String],
                             labelColumn: String,
                             nPartitions: Int = 100,
                             numTopFeatures: Int = 20, 
                             allVectorsDense: Boolean = true,
                             padded: Int = 0 /* if minimum value is negative */): InfoThSelectorModel = {
    val featureAssembler = new VectorAssembler()
      .setInputCols(inputCols)
      .setOutputCol("features")
    val processedDf = featureAssembler.transform(dataframe).select(labelColumn + INDEX_SUFFIX, "features")
    
    /** InfoSelector requires all vectors from the same type (either be sparse or dense) **/
    val rddData = processedDf.rdd.map {
        case Row(label: Double, features: Vector) =>
          val standardv = if(allVectorsDense){
            Vectors.dense(features.toArray.map(_ + padded))
          } else {
              val sparseVec = features.toSparse
              val newValues: Array[Double] = sparseVec.values.map(_ + padded)
              Vectors.sparse(sparseVec.size, sparseVec.indices, newValues)
          }
          
          Row.fromSeq(Seq(label, standardv))
      }
    
    val inputData = sqlContext.createDataFrame(rddData, processedDf.schema)
      
    val selector = new InfoThSelector()
        .setSelectCriterion("mrmr")
        .setNPartitions(nPartitions)
        .setNumTopFeatures(numTopFeatures)
        .setFeaturesCol("features")// this must be a feature vector
        .setLabelCol(labelColumn + INDEX_SUFFIX)
        .setOutputCol("selectedFeatures")

    selector.fit(inputData)
  }


  /**
    * The label column will have null values replaced with MISSING values in this case.
    * @return the feature selector fit to the data given the specified features to bin and label use as target.
    */
  def getSelectorModel(sqlContext: SQLContext, dataframe: DataFrame, inputCols: Array[String],
                          labelColumn: String,
                             nPartitions: Int = 100,
                             numTopFeatures: Int = 20,
                             allVectorsDense: Boolean = true,
                             padded: Int = 0): InfoThSelectorModel = {
    val processedDf = cleanLabelCol(dataframe, labelColumn)
    createSelectorModel(sqlContext, processedDf, inputCols, labelColumn, 
        nPartitions, numTopFeatures, allVectorsDense, padded)
  }


  def cleanLabelCol(dataframe: DataFrame, labelColumn: String): DataFrame = {
    val df = dataframe
      .withColumn(labelColumn + CLEAN_SUFFIX, when(col(labelColumn).isNull, lit(MISSING)).otherwise(col(labelColumn)))

    convertLabelToIndex(df, labelColumn + CLEAN_SUFFIX, labelColumn + INDEX_SUFFIX)
  }

  def cleanNumericCols(dataframe: DataFrame, numericCols: Array[String]): DataFrame = {
    var df = dataframe
    numericCols.foreach(column => {
      df = df.withColumn(column + CLEAN_SUFFIX, when(col(column).isNull, lit(Double.NaN)).otherwise(col(column)))
    })
    df
  }

  def convertLabelToIndex(df: DataFrame, inputCol: String, outputCol: String): DataFrame = {

    val labelIndexer = new StringIndexer()
      .setInputCol(inputCol)
      .setOutputCol(outputCol).fit(df)

    labelIndexer.transform(df)
  }

  def createSparkContext() = {
    // the [n] corresponds to the number of worker threads and should correspond ot the number of cores available.
    val conf = new SparkConf().setAppName("test-spark").setMaster("local[4]")
    // Changing the default parallelism gave slightly different results and did not do much for performance.
    //conf.set("spark.default.parallelism", "2")
    val sc = new SparkContext(conf)
    LogManager.getRootLogger.setLevel(Level.WARN)
    sc
  }
  
  /** @return standard csv data from the repo.
    */
  def readCSVData(sqlContext: SQLContext, file: String): DataFrame = {
       val df = sqlContext.read
        .format("com.databricks.spark.csv")
        .option("header", "true") // Use first line of all files as header
        .option("inferSchema", "true") // Automatically infer data types
        .load(FILE_PREFIX + file)
       df
  }
  
  /** @return dataset with 3 double columns. The first is the label column and contain null.
    */
  def readNullLabelTestData(sqlContext: SQLContext): DataFrame = {
    val data = SPARK_CTX.textFile(FILE_PREFIX + "null_label_test.data")
    val nullable = true

    val schema = StructType(List(
      StructField("label_IDX", DoubleType, nullable),
      StructField("col1", DoubleType, nullable),
      StructField("col2", DoubleType, nullable)
    ))
    // ints and dates must be read as doubles
    val rows = data.map(line => line.split(",").map(elem => elem.trim))
      .map(x => {Row.fromSeq(Seq(asDouble(x(0)), asDouble(x(1)), asDouble(x(2))))})

    sqlContext.createDataFrame(rows, schema)
  }

  private def asDateDouble(isoString: String) = {
    if (isoString == NULL_VALUE) Double.NaN
    else ISO_DATE_FORMAT.parseDateTime(isoString).getMillis.toString.toDouble
  }

  // label cannot currently have null values - see #8.
  private def asString(value: String) = if (value == NULL_VALUE) null else value
  private def asDouble(value: String) = if (value == NULL_VALUE) Double.NaN else value.toDouble
}
