package org.apache.spark.datagen;

import scala.collection.mutable
import scala.util.Random

/**
 * Generates a random decision tree avoiding non-sense ones.
 */
class RandomDecisionTreeGenerator(
    val nFeatures: Int,
    val maxBins: Int,
    val maxDepth: Int,
    val nLabels: Int,
    val randomSeed: Long = System.currentTimeMillis()) {

  private val probDecision = 0.10
  private val LT = true
  private val GE = false
  private case class Restriction(variable: Int, comparison: Boolean, threshold: Double)
  private val maxTrials: (Int => Int) = depth => 1000 /*if (depth == 1) Int.MaxValue else 1000/depth*/

  def generateTree = {

    val deciderStack = new mutable.Stack[DecisionTree]
    val restrictionStack = new mutable.Stack[Restriction]
    val random = new Random(randomSeed)

    val variable = random.nextInt(nFeatures)
    val threshold = random.nextInt(maxBins).toDouble
    deciderStack.push(null)
    deciderStack.push(Decider(variable, threshold, null, null))
    var depth = 1

    while (depth > 0) {
      val dt = deciderStack.pop

      dt match {
        case Decider(variable, threshold, null, _) =>
          restrictionStack.push(Restriction(variable, LT, threshold))
          if (depth < maxDepth && random.nextDouble > probDecision) {
            var trials = 0
            val variableRestrictions = restrictionStack.filter(_.variable == variable)
            while (trials < maxTrials(depth) && trials >= 0) {
              trials += 1
              val variable = random.nextInt(nFeatures)
              val threshold = (random.nextInt(maxBins-1) + 1).toDouble

              var compatible = true
              for (res <- variableRestrictions if compatible) {
                compatible &&= (res match {
                  case Restriction(_, LT, th) if th <= threshold => false
                  case Restriction(_, GE, th) if th >= threshold => false
                  case _ => true
                })
              }

              if (compatible) {
                deciderStack.push(dt)
                deciderStack.push(Decider(variable, threshold, null, null))
                depth += 1
                trials = -1
              }
            }
            if (trials >= maxTrials(depth)) {
              deciderStack.push(Decider(variable, threshold, Decision(random.nextInt(nLabels)), null))
              restrictionStack.pop
            }
          } else {
            deciderStack.push(Decider(variable, threshold, Decision(random.nextInt(nLabels)), null))
            restrictionStack.pop
          }

        case Decider(variable, threshold, leftDecision, null) =>
          restrictionStack.push(Restriction(variable, GE, threshold))
          if (depth < maxDepth && random.nextDouble > probDecision) {
            var trials = 0
            val variableRestrictions = restrictionStack.filter(_.variable == variable)
            while (trials < maxTrials(depth) && trials >= 0) {
              trials += 1
              val variable = random.nextInt(nFeatures)
              val threshold = random.nextInt(maxBins).toDouble

              var compatible = true
              for (res <- variableRestrictions if compatible) {
                compatible &&= (res match {
                  case Restriction(_, LT, th) if th <= threshold => false
                  case Restriction(_, GE, th) if th >= threshold => false
                  case _ => true
                })
              }

              if (compatible) {
                deciderStack.push(dt)
                deciderStack.push(Decider(variable, threshold, null, null))
                depth += 1
                trials = -1
              }
            }
            if (trials >= maxTrials(depth)) {
              deciderStack.push(Decider(variable, threshold, leftDecision, Decision(random.nextInt(nLabels))))
              restrictionStack.pop
            }
          } else {
            deciderStack.push(Decider(variable, threshold, leftDecision, Decision(random.nextInt(nLabels))))
            restrictionStack.pop
          }

        case Decider(_, _, leftDecision, _) =>
          val lastDt = deciderStack.pop
          lastDt match {
            case Decider(variable, threshold, null, null) =>
              deciderStack.push(Decider(variable, threshold, dt, null))
              restrictionStack.pop
              depth -= 1
            case Decider(variable, threshold, leftDecision, null) =>
              deciderStack.push(Decider(variable, threshold, leftDecision, dt))
              restrictionStack.pop
              depth -= 1
            case null =>
              deciderStack.push(dt)
              depth -= 1
          }
      }

    }

    deciderStack.pop

  }

}
