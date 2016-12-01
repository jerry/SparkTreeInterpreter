package treeinterpreter.shadowforest

case class LeafNode(shadowNode: ShadowNode, steps: Array[TraversalStep]) {
  def impurityStats: Array[Double] = {
    shadowNode.impurityStats
  }
}
