package org.apache.spark.datagen;

import breeze.stats.distributions.Rand
import org.apache.spark.mllib.linalg.Vectors
import breeze.linalg._
import breeze.numerics.abs
import org.apache.spark.SparkContext
import breeze.stats.distributions.MultivariateGaussian
import org.apache.spark.mllib.regression.LabeledPoint
import scala.util.Random

/**
 * Generates continuous data from two multivariate Gaussians.
 */
class GaussianGenerator {

  def generate(
      sc: SparkContext,
      nPatterns: Long,
      nRel: Int,
      nRed: Int,
      nRand: Int,
      maxNorm: Double,
      minSpan: Double,
      maxSpan: Double,
      redNoise: Double) = {

    println("Generate Gaussian dataset.")

    val dataPointsPerPartition = nPatterns / sc.defaultParallelism

    // Generate mean vectors
    var basis: DenseMatrix[Double] = DenseMatrix.rand[Double](nRel, nRel, Rand.gaussian)
    basis = orth(basis)
    basis :*= maxNorm

    var mean1: DenseVector[Double] = basis * DenseVector.rand[Double](nRel, Rand.gaussian)
    var mean2: DenseVector[Double] = basis * DenseVector.rand[Double](nRel, Rand.gaussian)
    var correct = false
    while (!correct) {
      correct = true
      // resize vectors if they excede maxNorm
      val mean1Norm: Double = norm(mean1)
      if (mean1Norm > maxNorm) {
        println("Resizing means1")
        mean1 *= (maxNorm / mean1Norm)
      }
      val mean2Norm: Double = norm(mean2)
      if (mean2Norm > maxNorm) {
        println("Resizing means2")
        mean2 *= (maxNorm / mean2Norm)
      }

      // recalculate vectors if they are too close
      val distance: Double = norm(mean1 - mean2)
      if (distance < 2 * maxSpan) {
        println("Means too close, recalculating...")
        mean1 = basis * DenseVector.rand[Double](nRel, Rand.gaussian)
        mean2 = basis * DenseVector.rand[Double](nRel, Rand.gaussian)
        correct = false
      } else {
        correct = true
      }
    }

    // Generate covariance matrix
    val cov: DenseMatrix[Double] =
        diag(DenseVector.rand[Double](nRel, Rand.uniform.map(minSpan + _ * (maxSpan - minSpan))))

    // Generate transformation matrix to generate redundant features
    val basis1: DenseMatrix[Double] = DenseMatrix.rand[Double](nRel, nRel, Rand.gaussian)
    val U: DenseMatrix[Double] = orth(basis1)
    val basis2: DenseMatrix[Double] = DenseMatrix.rand(nRed, nRel, Rand.gaussian)
    val V: DenseMatrix[Double] = orth(basis2)
    val D: DenseMatrix[Double] = diag(DenseVector.rand[Double](nRel, Rand.uniform) + 1.0)
    val M: DenseMatrix[Double] = V * (D * U.t)

    // Multivariate Gaussian generators
    val gens = Array(new MultivariateGaussian(mean1, cov), new MultivariateGaussian(mean2, cov))

    // Generate data
    val rdd = sc.parallelize(Seq.empty[LabeledPoint], sc.defaultParallelism)
    val bcM = sc.broadcast(M)

    rdd.mapPartitions({ case _ =>
      val data = for (i <- (1L to dataPointsPerPartition)) yield {
        val label = if (i < dataPointsPerPartition / 2) 0 else 1
        val rel: DenseVector[Double] = gens(label).draw()
        val red: DenseVector[Double] = bcM.value * rel
        val random = new Random
        for (j <- 0 until red.length if (random.nextDouble < redNoise)) {
          red(j) = random.nextDouble
        }
        val rand = Array.fill[Double](nRand)(random.nextDouble)
        LabeledPoint(label.toDouble, Vectors.dense(rel.toArray ++ red.toArray ++ rand))
      }
      data.iterator
    })

  }

  /**
   * Calculates an orthonormal basis for the range of a according to the Gram-Schmidt algorithm.
   * @param a Matrix from which to calculate basis.
   * @return Orthonormal basis for the range of a
   */
  def orth(a: DenseMatrix[Double]) = {
    val m = a.rows
    val n = a.cols
    var w = a(::, 0)
    var b = DenseMatrix.create[Double](m, 1, (w / norm(w)).toArray)
    for (i <- 1 until n) {
      w = a(::, i)
      w = w - b * (b.t * w)
      b = DenseMatrix.horzcat(b, DenseMatrix.create[Double](m, 1, w.toArray))
    }
    b
  }


}
