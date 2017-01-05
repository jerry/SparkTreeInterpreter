package org.apache.spark.ml.tree

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.tree.impurity.ImpurityCalculator

class TransparentNode(node: Node, numClasses: Int, features: Vector, rootNode: Boolean = false, contributions: Option[Array[FeatureContribution]] = None) {
  val prediction: Double = node.prediction
  val impurityStats: ImpurityCalculator = node.impurityStats

  def featureContributions(): Option[Array[FeatureContribution]] = {
    contributions
  }


  def probabilityByLabelIndex(): Array[Double] = {
    (0 until numClasses).toArray.map(f => { node.impurityStats.prob(f.toDouble) })
  }

  def interpretationImpl(): TransparentNode = {
    if (rootNode) {
//      println(s"RootNode with ${node.numDescendants} descendants: prediction ${node.prediction}")
//      println(s"Bias: [${probabilityByLabelIndex.mkString(", ")}]")
    }

    if (node.subtreeDepth.equals(0)) { // My node is a leaf node. We have reached the bottom of the tree.
//      println("Leaf Node!")
//      println(s"Probability by Label Index: [${probabilityByLabelIndex().mkString(", ")}]")
      this
    } else {
      val n = node.asInstanceOf[InternalNode]
      val fIndex: Int = n.split.featureIndex

//      println(s"Internal node (subtreeDepth: ${node.subtreeDepth}, featureIndex: $fIndex)")
//      println(s"Probability by Label Index: [${probabilityByLabelIndex().mkString(", ")}]")

      val nextNode: Node = if (n.split.shouldGoLeft(features)) {
//        println(s"...going left for $fIndex")
        n.leftChild
      } else {
//        println(s"...going right for $fIndex")
        n.rightChild
      }

      val predictionDelta: Array[Double] = (0 until numClasses).toArray.map(f => {
        val oldProbForClass = impurityStats.prob(f.toDouble)
        val newProbForClass = nextNode.impurityStats.prob(f.toDouble)
        newProbForClass - oldProbForClass
      })

      // How much does the probability per class change due to this split?
      val contribution = FeatureContribution(fIndex, predictionDelta)
//      println(contribution.toString)

      val allContributions: Option[Array[FeatureContribution]] = if (contributions.isDefined) {
        Some(contributions.get ++ Array(contribution))
      } else {
        Some(Array(contribution))
      }

      new TransparentNode(nextNode, numClasses, features, false, allContributions).interpretationImpl()
    }
  }
}
