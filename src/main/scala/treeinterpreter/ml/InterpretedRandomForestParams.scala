package treeinterpreter.ml

import org.apache.spark.ml.tree._

private[treeinterpreter] trait InterpretedRandomForestParams extends TreeEnsembleParams
  with HasFeatureSubsetStrategy with HasNumTrees

private[treeinterpreter] object InterpretedRandomForestParams {
  // These options should be lowercase.
  final val supportedFeatureSubsetStrategies: Array[String] =
    Array("auto", "all", "onethird", "sqrt", "log2").map(_.toLowerCase)
}

private[treeinterpreter] trait InterpretedRandomForestClassifierParams
  extends InterpretedRandomForestParams with TreeClassifierParams

private[treeinterpreter] trait InterpretedRandomForestClassificationModelParams extends TreeEnsembleParams
  with HasFeatureSubsetStrategy with TreeClassifierParams

private[treeinterpreter] trait InterpretedRandomForestRegressorParams
  extends InterpretedRandomForestParams with TreeRegressorParams

private[treeinterpreter] trait InterpretedRandomForestRegressionModelParams extends TreeEnsembleParams
  with HasFeatureSubsetStrategy with TreeRegressorParams