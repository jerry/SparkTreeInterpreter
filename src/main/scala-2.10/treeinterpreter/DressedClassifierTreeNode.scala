package treeinterpreter

import org.apache.spark.ml.tree._
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.tree.configuration.FeatureType
import treeinterpreter.TreeInterpretationTools.NodeID

case class DressedClassifierTreeNode(node: Node) {
  val isLeaf: Boolean = node.toString.contains("LeafNode")

  val value: Double = node.prediction

  val id: Int = hashCode()
  val NodeID: Int = id

  val feature: Option[Int] = if (isLeaf) {
    None
  } else {
    Option(internalNodeInstance.get.split.featureIndex)
  }

  def internalNodeInstance: Option[InternalNode] = if (isLeaf) {
    None
  } else {
    Option(node.asInstanceOf[InternalNode])
  }

  def rightNode: Option[Node] = if (isLeaf) {
    None
  } else {
    Option(internalNodeInstance.get.rightChild)
  }

  def leftNode: Option[Node] = if (isLeaf) {
    None
  } else {
    Option(internalNodeInstance.get.leftChild)
  }

  def split: Option[Split] = if (internalNodeInstance.isDefined) {
    Option(internalNodeInstance.get.split)
  } else {
    None
  }

  def predictLeaf(features: Vector)(implicit node2Treenode: Node => DressedClassifierTreeNode): NodeID = {
    if (isLeaf) {
      id
    } else {
      val n = internalNodeInstance.get

      val nSplitName: FeatureType.FeatureType = if (split.get.isInstanceOf[CategoricalSplit]) {
        FeatureType.Categorical
      } else {
        FeatureType.Continuous
      }

      val fIndex = split.get.featureIndex

      if (nSplitName == FeatureType.Continuous) {

        val threshold = split.get.asInstanceOf[ContinuousSplit].threshold

        if (features(fIndex) <= threshold) {
          leftNode.get.predictLeaf(features)
        } else {
          rightNode.get.predictLeaf(features)
        }
      } else {
        val categories = split.get.asInstanceOf[CategoricalSplit].leftCategories

        if (categories.contains(features(fIndex))) {
          leftNode.get.predictLeaf(features)
        } else {
          rightNode.get.predictLeaf(features)
        }
      }
    }
  }

  override def toString: String = Array(node.toString, node.prediction, feature).mkString("||", ",", "||")
}
