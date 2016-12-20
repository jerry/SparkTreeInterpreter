package treeinterpreter

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.rdd.RDD

class DressedForest(rf: RandomForestModel) extends Serializable {
  val trees: Array[DressedTree] = rf.trees.map(tree => DressedTree.trainInterpreter(tree))

  def interpret(lp: LabeledPoint): TreeInterpretation = {
    val treeValues: Array[TreeInterpretation] = trees.map { dressedTree =>
      dressedTree.interpret(lp.features)
    }

    val t1: TreeInterpretation = treeValues.reduce(_.plus(_))

    val (avgBias, avgPrediction) = (t1.bias / t1.treeCount, t1.prediction/t1.treeCount)

    val avgContributions = t1.contributions.mapValues(_/t1.treeCount).map(identity)

    val checkSum = avgBias + avgContributions.values.sum

    TreeInterpretation(avgBias, avgPrediction, avgContributions, t1.treeCount, checkSum)
  }

  def interpret(testSet: RDD[LabeledPoint]): RDD[TreeInterpretation] = {
    testSet.map(lp => {
      interpret(lp)
    })
  }
}
