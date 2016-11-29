package treeinterpreter

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.model.{DecisionTreeModel, RandomForestModel}
import org.apache.spark.rdd.RDD

object Interp {
  def interpretModel(rf: RandomForestModel, testSet: RDD[LabeledPoint]): RDD[TreeInterpretation] = {
    val dressedForest = new DressedForest(rf)
    dressedForest.interpret(testSet)
  }

  def interpretModel(model: DecisionTreeModel, testSet: RDD[LabeledPoint]): RDD[TreeInterpretation] = {
    val dressedTree = DressedTree.trainInterpreter(model)
    testSet.map(lp => dressedTree.interpret(lp.features))
  }
}
