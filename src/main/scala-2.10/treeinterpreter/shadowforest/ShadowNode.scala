package treeinterpreter.shadowforest

import org.apache.spark.ml.linalg.Vector

case class ShadowNode(id: Int, prediction: Double, impurity: Double, impurityStats: Array[Double], gain: Double, leftChild: Int, rightChild: Int, split: ShadowSplit) {
  def predictImpl(features: Vector): LeafNode = {
    // TODO: Traverse the tree and get the right node - this isn't it.
    LeafNode(this, Array())
  }
}
