//package treeinterpreter.ml
//
//
//import org.apache.spark.mllib.linalg.Vector
//
//private[treeinterpreter] abstract class Node extends Serializable {
//
//  def predict(features: Vector): Double
//}
//
//private[treeinterpreter] class InnerNode(val split: Split, val leftChild: Node, val rightChild: Node) extends Node {
//
//  override def predict(features: Vector): Double = {
//    if (split.shouldGoLeft(features)) {
//      leftChild.predict(features)
//    } else {
//      rightChild.predict(features)
//    }
//  }
//}
//
//private[treeinterpreter] class LeafNode(val prediction: Double) extends Node {
//
//  override def predict(features: Vector): Double = prediction
//}