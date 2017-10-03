package org.apache.spark.datagen;

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.regression.LabeledPoint
import scala.util.Random
import org.apache.spark.mllib.linalg.Vectors
import scala.collection.mutable.ArrayBuffer

/**
 * Generates discrete data
 */
object DiscreteDataGenerator extends Serializable {

  def generate(
      sc: SparkContext,
      nRelevantFeatures: Int,
      nDataPoints: Long,
      noiseOnRelevant: Double,
      redundantNoises: Seq[Double],
      nRandomFeatures: Int,
      maxBins: Int = 15,
      maxDepth: Int = 20,
      nLabels: Int = 2,
      seed: Long = System.currentTimeMillis(),
      outputPath: String = "src/test/resources/random.data") = {

    val generator =
      new RandomDecisionTreeGenerator(nRelevantFeatures, maxBins, maxDepth, nLabels, seed)
    val tree = generator.generateTree
    val bcTree = sc.broadcast(tree)

    var rdd = sc.parallelize(Seq.empty[LabeledPoint], sc.defaultParallelism)
    val dataPointsPerPartition = nDataPoints/sc.defaultParallelism

    val rnd = new Random(seed)
    val innerSeed = rnd.nextLong()

    // Generate relevant features
    rdd = rdd.mapPartitions({ case _ =>
      val random = new Random(innerSeed)
      val data = for (_ <- 1L to dataPointsPerPartition) yield {
        val features = Array.fill(nRelevantFeatures)(random.nextInt(maxBins).toDouble)
        var label = bcTree.value.decide(features)
        if (random.nextDouble() < noiseOnRelevant) {
          val oldLabel = label
          while (label == oldLabel) {
            label = random.nextInt(nLabels)
          }
        }
        LabeledPoint(label, Vectors.dense(features))
      }

      data.iterator
    })

    // Generate redundant features
    var redundantFeatures = new ArrayBuffer[Int]()
    for (noise <- redundantNoises) {
      val feat2Replicate = rnd.nextInt(nRelevantFeatures)
      redundantFeatures += feat2Replicate
      rdd = rdd.map({ case LabeledPoint(label, features) =>
        val random = new Random(innerSeed ^ features.hashCode)
        val oldValue = features.toArray(feat2Replicate)
        var newValue = oldValue
        if (random.nextDouble < noise) {
          while (oldValue == newValue) {
            newValue = random.nextInt(maxBins)
          }
        }
        LabeledPoint(label, Vectors.dense(features.toArray :+ newValue))
      })
    }
    println("Redundant features: " + redundantFeatures)


    // Generate random features
    val initRandom = rdd.first.features.size
    for (_ <- 1 to nRandomFeatures) {
      rdd = rdd.map({ case LabeledPoint(label, features) =>
        val random = new Random(innerSeed ^ features.hashCode)
        LabeledPoint(label, Vectors.dense(features.toArray :+ random.nextInt(maxBins).toDouble))
      })
    }
    println("Random features: " + (initRandom until rdd.first().features.size))

    rdd.saveAsTextFile(outputPath)
    rdd

  }

}