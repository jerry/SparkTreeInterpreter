package org.apache.spark.ml.tree

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.tree.impurity.ImpurityCalculator

/** Collects feature contribution data while traversing a DecisionTree
  *
  * @param node associated ML tree node
  * @param numClasses number of classes in the model - use 0 for a regression tree
  * @param features feature values for prediction
  * @param rootNode is this the tree's root node?
  * @param contributions contributions collected in the tree thus far
  */
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
      val n = node.asInstanceOf[InternalNode] // We know we are an internal node.
      val fIndex: Int = n.split.featureIndex

      val nextNode: Node = if (n.split.shouldGoLeft(features)) {
        n.leftChild
      } else {
        n.rightChild
      }

      // How much does the prediction change due to this split?
      val contributionArray = if (numClasses != 0) {
        Array(FeatureContribution(fIndex, predictionDelta(nextNode)))
      } else {
        Array(FeatureContribution(fIndex, Array(nextNode.prediction - prediction)))
      }

      val allContributions: Option[Array[FeatureContribution]] = if (contributions.isDefined) {
        Some(contributions.get ++ contributionArray)
      } else {
        Some(contributionArray)
      }

      new TransparentNode(nextNode, numClasses, features, false, allContributions).interpretationImpl()
    }
  }

  def predictionDelta(nextNode: Node): Array[Double] = {
    (0 until numClasses).toArray.map(f => {
      val oldProbForClass = impurityStats.prob(f.toDouble)
      val newProbForClass = nextNode.impurityStats.prob(f.toDouble)
      newProbForClass - oldProbForClass
    })
  }
}
