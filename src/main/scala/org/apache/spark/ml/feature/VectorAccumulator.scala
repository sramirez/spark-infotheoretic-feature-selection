package org.apache.spark.ml.feature


import breeze.linalg._
import breeze.numerics._
import org.apache.spark.util.AccumulatorV2
/**
 * @author sramirez
 */
class VectorAccumulator(val rows: Int) extends AccumulatorV2[Vector[Long], Vector[Long]] {
  
  def this(m: Vector[Long]) = {
    this(m.size)
    this.accVector = m.copy
  }

  private var accVector: Vector[Long] = Vector.zeros[Long](rows)
  private var zero: Boolean = true

  def reset(): Unit = {
    accVector = Vector.zeros[Long](rows)
    zero = true
  }

  def add(v: Vector[Long]): Unit = {
    if(isZero) 
      zero = false
    accVector += v
  }
  
  def isZero(): Boolean = zero
  
  def merge(other: AccumulatorV2[Vector[Long], Vector[Long]]): Unit = accVector += other.value
  
  def value: Vector[Long] = accVector
  
  def copy(): AccumulatorV2[Vector[Long], Vector[Long]] = new VectorAccumulator(accVector)
}