package org.apache.spark.ml.tree

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.tree.impurity.ImpurityCalculator

class TransparentNode(node: Node, numClasses: Int, features: Vector, rootNode: Boolean = false, contributions: Option[Array[FeatureContribution]] = None) {
  val prediction: Double = node.prediction
  val impurityStats: ImpurityCalculator = node.impurityStats

  def featureContributions(): Option[Array[FeatureContribution]] = {
    contributions
  }

  def interpretationImpl(): TransparentNode = {
    if (node.subtreeDepth.equals(0)) { // My node is a leaf node. We have reached the bottom of the tree.
      this
    } else {
      val n = node.asInstanceOf[InternalNode]
      val fIndex: Int = n.split.featureIndex

      val nextNode: Node = if (n.split.shouldGoLeft(features)) {
        n.leftChild
      } else {
        n.rightChild
      }

      val predictionDelta: Array[Double] = (0 until numClasses).toArray.map(f => {
        val oldProbForClass = impurityStats.prob(f.toDouble)
        val newProbForClass = nextNode.impurityStats.prob(f.toDouble)
        newProbForClass - oldProbForClass
      })

      // How much does the probability per class change due to this split?
      val contribution = FeatureContribution(fIndex, predictionDelta)

      val allContributions: Option[Array[FeatureContribution]] = if (contributions.isDefined) {
        Some(contributions.get ++ Array(contribution))
      } else {
        Some(Array(contribution))
      }

      new TransparentNode(nextNode, numClasses, features, false, allContributions).interpretationImpl()
    }
  }
}
