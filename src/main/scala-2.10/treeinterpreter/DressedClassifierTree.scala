package treeinterpreter

import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.tree._
import org.apache.spark.mllib.linalg.Vector
import treeinterpreter.TreeInterpretationTools._

case class DressedClassifierTree(model: DecisionTreeClassificationModel, bias: Double, contributionMap: NodeContributions) {
  val topTreeNode: Node = model.rootNode

  implicit def nodeType(node: Node): DressedClassifierTreeNode = DressedClassifierTreeNode(node)

  private def predictLeafID(features: Vector): NodeID = topTreeNode.predictLeaf(features)

  def predictLeaf(point: Vector): TreeInterpretation = {
    val leaf: NodeID = predictLeafID(point)
    val prediction: Double = 0.20 // model.predict(point)
    val contribution: NodeContribution = contributionMap(leaf)
    TreeInterpretation(bias, prediction, contribution)
  }

  def interpret(point: Vector): TreeInterpretation = predictLeaf(point)
}

object DressedClassifierTree {
  def trainInterpreter(model: DecisionTreeClassificationModel): DressedClassifierTree = {
    val topNode = model.rootNode
    val bias = topNode.prediction

    def buildPath(paths: ClassifierPath, node: Node): ClassifierPathBundle = {
      // Get the current node as ClassificationNode or a RegressionNode
      //val treeNode = TreeNode.typedNode(node, model.algo)
      val cNode = DressedClassifierTreeNode(node)

      // If this is a leaf node, then we're done on this path.
      if (cNode.isLeaf) Array(paths :+ cNode)
      else {
        // Otherwise, get the nodes to the left and right and recurse them, as well.
        val buildRight = buildPath(paths :+ cNode, cNode.rightNode.get)
        val buildLeft = buildPath(paths :+ cNode, cNode.leftNode.get)
        buildRight ++ buildLeft
      }
    }

    // Recurse the tree, collecting a PathBundle, in reverse order.
    val paths = buildPath(Array(), topNode).map(_.sorted.reverse)
    DressedTree.arrayprint(paths)

    // As we ascend the tree, capture the change in value and the feature id
    val contributions: NodeContributions = paths.flatMap(path => {
      val contribMap = {
        {
          path zip path.tail
        }.flatMap {
          case (currentNode, prevNode) =>
            Map(prevNode.feature -> {
              currentNode.value - prevNode.value
            })
        }
      }.foldLeft(Map[Feature, Double]())(_ + _)

      val leafID = path.head.hashCode()
      Map(leafID -> contribMap)
    }).toMap

    DressedClassifierTree(model, bias, contributions)
  }
}

