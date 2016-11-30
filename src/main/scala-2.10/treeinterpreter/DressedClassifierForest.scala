package treeinterpreter

import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, RandomForestClassificationModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

class DressedClassifierForest(rf: RandomForestClassificationModel) extends Serializable {
  val trees: Array[DressedClassifierTree] = decisionTrees().map(tree => DressedClassifierTree.trainInterpreter(tree))

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

  def decisionTrees(): Array[DecisionTreeClassificationModel] = {
    val ru = scala.reflect.runtime.universe
    val m = ru.runtimeMirror(getClass.getClassLoader)

    val reflectedForest = m.reflect(rf)
    val sym: ru.TermSymbol = ru.typeOf[RandomForestClassificationModel].declaration(ru.newTermName("_trees")).asTerm.accessed.asTerm
    val treesMirror = reflectedForest.reflectField(sym)
    treesMirror.get.asInstanceOf[Array[DecisionTreeClassificationModel]]
  }
}
