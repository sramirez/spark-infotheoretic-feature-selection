package org.apache.spark.ml.feature


import breeze.linalg._
import breeze.numerics._
import org.apache.spark.util.AccumulatorV2
/**
 * @author sramirez
 */
class MatrixAccumulator(val rows: Int, val cols: Int) extends AccumulatorV2[Matrix[Long], Matrix[Long]] {
  
  def this(m: Matrix[Long]) = {
    this(m.rows, m.cols)
    this.accMatrix = m.copy
  }

  private var accMatrix: Matrix[Long] = Matrix.zeros[Long](rows, cols)
  private var zero: Boolean = true

  def reset(): Unit = {
    accMatrix = Matrix.zeros[Long](rows, cols)
    zero = true
  }

  def add(v: Matrix[Long]): Unit = {
    if(isZero) 
      zero = false
    accMatrix += v
  }
  
  def isZero(): Boolean = zero
  
  def merge(other: AccumulatorV2[Matrix[Long], Matrix[Long]]): Unit = accMatrix += other.value
  
  def value: Matrix[Long] = accMatrix
  
  def copy(): AccumulatorV2[Matrix[Long], Matrix[Long]] = new MatrixAccumulator(accMatrix)
}