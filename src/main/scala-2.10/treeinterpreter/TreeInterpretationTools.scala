package treeinterpreter

import treeinterpreter.TreeNode.TreeNode

object TreeInterpretationTools {
  type NodeID = Int
  type NodeMap = Map[NodeID, Double]
  type ClassifierPath = Array[DressedClassifierTreeNode]
  type ClassifierPathBundle = Array[ClassifierPath]
  type Path = Array[TreeNode]
  type PathBundle = Array[Path]
  type Feature = Option[NodeID]
  type NodeContribution = Map[Feature, Double]
  type NodeContributions = Map[NodeID, NodeContribution]
}