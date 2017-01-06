package org.apache.spark.ml.tree

import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.tree.impl.RandomForest
import org.apache.spark.ml.util.{Instrumentation, MetadataUtils}
import org.apache.spark.mllib.tree.configuration.{Algo => OldAlgo}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset

class InterpretedRandomForestClassifier extends RandomForestClassifier {
  override protected def train(dataset: Dataset[_]): InterpretedRandomForestClassificationModel = {
    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))
    val numClasses: Int = getNumClasses(dataset)
    val oldDataset: RDD[LabeledPoint] = extractLabeledPoints(dataset, numClasses)
    val strategy =
      super.getOldStrategy(categoricalFeatures, numClasses, OldAlgo.Classification, getOldImpurity)

    val instr = Instrumentation.create(this, oldDataset)
    instr.logParams(params: _*)

    val trees = RandomForest
      .run(oldDataset, strategy, getNumTrees, getFeatureSubsetStrategy, getSeed, Some(instr))
      .map(_.asInstanceOf[DecisionTreeClassificationModel])

    val numFeatures = oldDataset.first().features.size
    val m = new InterpretedRandomForestClassificationModel(trees, numFeatures, numClasses)
    instr.logSuccess(m)
    m
  }
}