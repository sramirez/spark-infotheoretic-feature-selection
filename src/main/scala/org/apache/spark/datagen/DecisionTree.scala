package org.apache.spark.datagen;

/**
 * Abstract class for a Decision Tree
 */
sealed abstract class DecisionTree extends Serializable {
  
  def decide(data: Array[Double]): Int

  def stats: DTStats

}


/**
 * Discriminative node
 */
case class Decider(
    variable: Int,
    threshold: Double,
    leftDecision: DecisionTree,
    rightDecision: DecisionTree)
  extends DecisionTree{

  override def decide(data: Array[Double]): Int = {
    if (data(variable) < threshold) {
      leftDecision match {
        case Decision(label) => label
        case decider: Decider => decider.decide(data)
      }
    } else {
      rightDecision match {
        case Decision(label) => label
        case decider: Decider => decider.decide(data)
      }
    }
  }

  override def stats = {
    val leftStats = leftDecision.stats
    val rightStats = rightDecision.stats
    DTStats(leftStats.leafs + rightStats.leafs,
            math.max(leftStats.maxDepth, rightStats.maxDepth) + 1,
            math.min(leftStats.minDepth, rightStats.minDepth) + 1,
            leftStats.variables ++ rightStats.variables + variable)
  }
}


/**
 * Leaf node.
 */
case class Decision(label: Int) extends DecisionTree {

  override def decide(data: Array[Double]): Int = label

  override def stats = DTStats(1, 0, 0, Set.empty[Int])

}

/**
 * Case class used for getting Decision Tree Statistics.
 */
case class DTStats(leafs: Long, maxDepth: Int, minDepth: Int, variables: Set[Int])
