package org.apache.spark.ml.tree

import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, RandomForestClassificationModel}
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.util.{Identifiable, MLReadable, MLReader}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, Dataset}

class InterpretedRandomForestClassificationModel (override val uid: String,
                                                 private val _trees: Array[DecisionTreeClassificationModel],
                                                 override val numFeatures: Int,
                                                 override val numClasses: Int)
  extends RandomForestClassificationModel(uid: String, _trees: Array[DecisionTreeClassificationModel], numFeatures: Int, numClasses: Int) {
  private[ml] def this(trees: Array[DecisionTreeClassificationModel], numFeatures: Int, numClasses: Int) =
    this(Identifiable.randomUID("rfc"), trees, numFeatures, numClasses)

  override def toString: String = "InterpretedRandomForestClassificationModel"

  override protected def transformImpl(dataset: Dataset[_]): DataFrame = {
    println(dataset.schema.treeString)

    val bcastModel = dataset.sparkSession.sparkContext.broadcast(this)
    val predictUDF = udf { (features: Any) =>
      // Should be a vector with numFeatures + 3 elements
      // bcastModel.value.predict(features.asInstanceOf[Vector])
      bcastModel.value.interpretedPrediction(features.asInstanceOf[Vector])
    }

    //println(s"setting ${predictionCol.name}")
    val d = dataset.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
    println(d.schema.treeString)
    d
  }

  /**
    * Transforms dataset by reading from [[featuresCol]], and appending new columns as specified by
    * parameters:
    *  - predicted labels as [[predictionCol]] of type [[Double]]
    *  - raw predictions (confidences) as [[rawPredictionCol]] of type `Vector`
    *  - probability of each class as [[probabilityCol]] of type `Vector`.
    *
    * @param dataset input dataset
    * @return transformed dataset
    */
  override def transform(dataset: Dataset[_]): DataFrame = {
    var outputData = dataset
    val predictRawUDF = udf { (features: Any) =>
      interpretedPrediction(features.asInstanceOf[Vector])
    }
    outputData = outputData.withColumn("interpretedPrediction", predictRawUDF(col(getFeaturesCol)))

    val extractPredictionElementUDF = udf { (interpretedPrediction: Any) =>
      extractPredictionElement(interpretedPrediction.asInstanceOf[Vector], 0)
    }
    outputData = outputData.withColumn("prediction", extractPredictionElementUDF(col("interpretedPrediction")))

    val extractPredictedLabelProbabilityUDF = udf { (interpretedPrediction: Any) =>
      extractPredictionElement(interpretedPrediction.asInstanceOf[Vector], 1)
    }
    outputData = outputData.withColumn("predictedLabelProbability", extractPredictedLabelProbabilityUDF(col("interpretedPrediction")))

    val extractBiasUDF = udf { (interpretedPrediction: Any) =>
      extractPredictionElement(interpretedPrediction.asInstanceOf[Vector], 2)
    }
    outputData = outputData.withColumn("bias", extractBiasUDF(col("interpretedPrediction")))

    val extractChecksumUDF = udf { (interpretedPrediction: Any) =>
      extractPredictionElement(interpretedPrediction.asInstanceOf[Vector], 3)
    }
    outputData = outputData.withColumn("checksum", extractChecksumUDF(col("interpretedPrediction")))

    val extractProbabilitiesByClassUDF = udf { (interpretedPrediction: Any) =>
      extractProbabilitiesByClass(interpretedPrediction.asInstanceOf[Vector])
    }
    outputData = outputData.withColumn("probabilitiesByClass", extractProbabilitiesByClassUDF(col("interpretedPrediction")))

    val extractContributionsUDF = udf { (interpretedPrediction: Any) =>
      extractContributions(interpretedPrediction.asInstanceOf[Vector])
    }
    outputData = outputData.withColumn("contributions", extractContributionsUDF(col("interpretedPrediction")))

    // Array(prediction, predictedLabelProbability, avgBias, checkSum) ++ probabilities.toArray ++ averagedContributions
    // TODO: extract the predictedLabel (string?), probabilities by class (Vector), bias (double) and feature contributions (Vector)
    outputData.toDF
  }

  def extractPredictionElement(interpretedPrediction: Vector, elementId: Int): Double = {
    interpretedPrediction(elementId)
  }

  def extractProbabilitiesByClass(interpretedPrediction: Vector): Vector = {
    Vectors.dense(interpretedPrediction.toArray.view(4, numClasses + 4).toArray)
  }

  def extractContributions(interpretedPrediction: Vector): Vector = {
    Vectors.dense(interpretedPrediction.toArray.view(numClasses + 4, numFeatures + numClasses + 4).toArray)
  }

  /** Contributions are the contributions to a given prediction by the ensemble of decision trees, where each represents the change in probability of the
    * predicted label at each step in each decision tree.
    *
    * comes in as 1 per class, per feature, per tree
    * leaves as an array of doubles (1 per class), per feature
    *
    */
  def averageContributions(contributions: Array[Option[Array[FeatureContribution]]], features: Vector): Array[Array[Double]] = {
    val allContributions: Array[FeatureContribution] = contributions.flatMap(f => f.getOrElse(new Array[FeatureContribution](numClasses)))
    val avgContributions: Array[Array[Double]] = (0 until features.size).toArray.map(f => {
      // get the per-class contributions for the current featureIndex, f
      val i: Array[FeatureContribution] = allContributions.filter(p => p.featureIndex.equals(f))
      val contributionCount = getNumTrees

      (0 until numClasses).toArray.map(j => {
        val sumForFeature = i.map(f => f.contribution(j)).sum
        val avgForFeature = if (sumForFeature.equals(0.0) || contributionCount.equals(0.0)) {
          0.0
        } else {
          sumForFeature / contributionCount
        }
        avgForFeature
      })
    })
    avgContributions
  }

  def interpretedPrediction(features: Array[Double]): Vector = {
    interpretedPrediction(Vectors.dense(features))
  }

  /** Get a prediction, with bias, checksum, and, per-feature contribution amounts
    *
    * @param features vector of feature values
    * @return prediction, bias, checksum, contribution amounts
    */
  def interpretedPrediction(features: Vector): Vector = {
    // Classifies using majority votes.
    // Ignore the tree weights since all are 1.0 for now.
    val votes = Array.fill[Double](numClasses)(0.0)

    // per tree, per prediction, the impact of each internal node on the final prediction
    val treeContributions = new Array[Option[Array[FeatureContribution]]](getNumTrees)
    val biasValues = new Array[Array[Double]](getNumTrees)
    var nextTree: Int = 0

    _trees.view.foreach { tree =>
      val rootNode = tree.rootNode
      val transparentLeafNode: TransparentNode = new TransparentNode(rootNode, numClasses, features, rootNode = true).interpretationImpl()

      val fc = transparentLeafNode.featureContributions()
      treeContributions.update(nextTree, fc)

      biasValues.update(nextTree, rootNode.impurityStats.stats)
      nextTree += 1

      val classCounts: Array[Double] = transparentLeafNode.impurityStats.stats
      val total = classCounts.sum
      if (total != 0) {
        var i = 0
        while (i < numClasses) {
          votes(i) += classCounts(i) / total
          i += 1
        }
      }
    }

    val rawPrediction = Vectors.dense(votes)
    val probabilities: Vector = raw2probability(rawPrediction)
    val prediction: Double = if (!isDefined(thresholds)) {
      rawPrediction.argmax
    } else {
      probability2prediction(probabilities)
    }
    val predictionLabelIndex: Int = prediction.toInt


    val allAveragedContributions: Array[Array[Double]] = averageContributions(treeContributions, features) // Get the contributions for each possible label
    assert(allAveragedContributions.length.equals(numFeatures))

    // Use the contributions from the actual label
    val averagedContributions: Array[Double] = allAveragedContributions.map(b => b(prediction.toInt))
    assert(averagedContributions.length.equals(numFeatures))


    val avgBias: Double = biasValues.map(f => {
      f(predictionLabelIndex) / f.sum
    }).sum / getNumTrees.toDouble
    val predictedLabelProbability = probabilities(predictionLabelIndex)

    val checkSum: Double = averagedContributions.sum + avgBias

    val x: Array[Double] = Array(prediction, predictedLabelProbability, avgBias, checkSum) ++ probabilities.toArray ++ averagedContributions

    Vectors.dense(x)
  }
}

object InterpretedRandomForestClassificationModel extends MLReadable[InterpretedRandomForestClassificationModel] {
  override def read: MLReader[InterpretedRandomForestClassificationModel] =
    new InterpretedRandomForestClassificationModelReader
  override def load(path: String): InterpretedRandomForestClassificationModel = super.load(path)
}
