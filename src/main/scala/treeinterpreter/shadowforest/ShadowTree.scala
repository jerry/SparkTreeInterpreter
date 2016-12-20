package treeinterpreter.shadowforest

import scala.collection.mutable.ArrayBuffer

case class ShadowTree(treeID: Int, metadata: String, weights: Double) {
  var nodes: ArrayBuffer[ShadowNode] = new ArrayBuffer

  def bias: Double = {
    rootNode.prediction
  }

  def rootNode: ShadowNode = {
    nodes(0) // TODO: Make sure this is really the root node
  }

  def nodeCount(): Int = {
    nodes.length
  }

  def addNode(node: ShadowNode): Unit = {
    nodes.append(node)
  }
}
