package treeinterpreter

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.tree.model.{DecisionTreeModel, Node}
import treeinterpreter.TreeInterpretationTools._
import treeinterpreter.TreeNode._

case class DressedTree(model: DecisionTreeModel, bias: Double, contributionMap: NodeContributions) {
  val topTreeNode: Node = model.topNode

  implicit def nodeType(node: Node): TreeNode = TreeNode.typedNode(node, model.algo)

  private def predictLeafID(features: Vector): NodeID = topTreeNode.predictLeaf(features)

  def predictLeaf(point: Vector): TreeInterpretation = {
    val leaf: NodeID = predictLeafID(point)
    val prediction: Double = model.predict(point)
    val contribution: NodeContribution = contributionMap(leaf)
    TreeInterpretation(bias, prediction, contribution)
  }

  def interpret(point: Vector): TreeInterpretation = predictLeaf(point)
}

object DressedTree {
  def arrayprint[A](x: Array[A]): Unit = println(x.deep.mkString("\n"))

  def trainInterpreter(model: DecisionTreeModel): DressedTree = {
    val topNode = model.topNode

    val bias = TreeNode.typedNode(model.topNode, model.algo).value

    def buildPath(paths: Path, node: Node): PathBundle = {
      // Get the current node as ClassificationNode or a RegressionNode
      val treeNode = TreeNode.typedNode(node, model.algo)

      // If this is a leaf node, then we're done on this path.
      if (node.isLeaf) Array(paths :+ treeNode)
      else {
        // Otherwise, get the nodes to the left and right and recurse them, as well.
        import node._
        val buildRight = buildPath(paths :+ treeNode, rightNode.get)
        val buildLeft = buildPath(paths :+ treeNode, leftNode.get)
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
              currentNode.value -prevNode.value
            })
        }
      }.foldLeft(Map[Feature, Double]())(_ + _)

      val leafID = path.head.NodeID
      Map(leafID -> contribMap)
    }).toMap

    DressedTree(model, bias, contributions)
  }
}
