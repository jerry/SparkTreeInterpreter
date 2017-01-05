package org.apache.spark.ml.tree

import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, ProbabilisticClassificationModel, RandomForestClassificationModel}
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, Dataset}

class InterpretedRandomForestClassificationModel private[ml] (override val uid: String,
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
    //    transformSchema(dataset.schema, logging = true)
    //    if (isDefined(thresholds)) {
    //      require($(thresholds).length == numClasses, this.getClass.getSimpleName +
    //        ".transform() called with non-matching numClasses and thresholds.length." +
    //        s" numClasses=$numClasses, but thresholds has length ${$(thresholds).length}")
    //    }
    //
    //    // Output selected columns only.
    //    // This is a bit complicated since it tries to avoid repeated computation.
    //    var outputData = dataset
    //    var numColsOutput = 0
    //    if ($(rawPredictionCol).nonEmpty) {
    //      val predictRawUDF = udf { (features: Any) =>
    //        predictRaw(features.asInstanceOf[FeaturesType])
    //      }
    //      outputData = outputData.withColumn(getRawPredictionCol, predictRawUDF(col(getFeaturesCol)))
    //      numColsOutput += 1
    //    }
    //    if ($(probabilityCol).nonEmpty) {
    //      val probUDF = if ($(rawPredictionCol).nonEmpty) {
    //        udf(raw2probability _).apply(col($(rawPredictionCol)))
    //      } else {
    //        val probabilityUDF = udf { (features: Any) =>
    //          predictProbability(features.asInstanceOf[FeaturesType])
    //        }
    //        probabilityUDF(col($(featuresCol)))
    //      }
    //      outputData = outputData.withColumn($(probabilityCol), probUDF)
    //      numColsOutput += 1
    //    }
    //    if ($(predictionCol).nonEmpty) {
    //      val predUDF = if ($(rawPredictionCol).nonEmpty) {
    //        udf(raw2prediction _).apply(col($(rawPredictionCol)))
    //      } else if ($(probabilityCol).nonEmpty) {
    //        udf(probability2prediction _).apply(col($(probabilityCol)))
    //      } else {
    //        val predictUDF = udf { (features: Any) =>
    //          predict(features.asInstanceOf[FeaturesType])
    //        }
    //        predictUDF(col($(featuresCol)))
    //      }
    //      outputData = outputData.withColumn($(predictionCol), predUDF)
    //      numColsOutput += 1
    //    }
    //
    //    if (numColsOutput == 0) {
    //      this.logWarning(s"$uid: ProbabilisticClassificationModel.transform() was called as NOOP" +
    //        " since no output columns were set.")
    //    }
    //    outputData.toDF
    var outputData = dataset
    val predictRawUDF = udf { (features: Any) =>
      interpretedPrediction(features.asInstanceOf[Vector])
    }
    outputData = outputData.withColumn("interpretedPrediction", predictRawUDF(col(getFeaturesCol)))
    outputData.toDF
  }

  /** Contributions are the contributions to a given prediction by the ensemble of decision trees, where each represents the change in probability of the
    * predicted label at each step in each decision tree.
    *
    * comes in as 1 per class, per feature, per tree
    * leaves as an array of doubles (1 per class), per feature
    *
    */
  def averageContributions(contributions: Array[Option[Array[FeatureContribution]]], features: Vector): Array[Array[Double]] = {
    // TODO: flatMap contributions, filter and average per feature index and class
    val allContributions: Array[FeatureContribution] = contributions.flatMap(f => f.getOrElse(new Array[FeatureContribution](numClasses)))
    val avgContributions: Array[Array[Double]] = (0 until features.size).toArray.map(f => {
      // get the per-class contributions for the current featureIndex, f
      val i: Array[FeatureContribution] = allContributions.filter(p => p.featureIndex.equals(f))
      val contributionCount = getNumTrees //i.length.toDouble

      (0 until numClasses).toArray.map(j => {
        val sumForFeature = i.map(f => f.contribution(j)).sum
        val avgForFeature = if (sumForFeature.equals(0.0) || contributionCount.equals(0.0)) {
          0.0
        } else {
          sumForFeature/contributionCount
        }
        // println(s"featureIndex: $f - $sumForFeature/$contributionCount = $avgForFeature for label $j")
        avgForFeature
      })
    })
    avgContributions
  }

  def interpretedPrediction(features: Array[Double]): Vector = {
    interpretedPrediction(Vectors.dense(features))
  }

  /** Get a prediction, with bias, checksum, and, per feature, contribution amounts
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
      //println(s"\n---- starting tree $nextTree ----")
      val rootNode = tree.rootNode
      val transparentLeafNode: TransparentNode = new TransparentNode(tree.rootNode, numClasses, features, rootNode = true).interpretationImpl()

      val fc = transparentLeafNode.featureContributions()
      treeContributions.update(nextTree, fc)

      // println(s"bias per class at root node: [${tree.rootNode.impurityStats.stats.mkString(", ")}]")
      biasValues.update(nextTree, tree.rootNode.impurityStats.stats)
      //println(s"---- finishing tree $nextTree ----\n")
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

    // Get the contributions for each possible label
    val allAveragedContributions: Array[Array[Double]] = averageContributions(treeContributions, features)
//    println(s"allAveragedContributions - ${allAveragedContributions.length}, numClasses - $numClasses")
    assert(allAveragedContributions.length.equals(numFeatures))

    // Use the contributions from the actual label
    val averagedContributions: Array[Double] = allAveragedContributions.map(b => b(prediction.toInt))
//    println(averagedContributions.length)
//    println(features.size)
    assert(averagedContributions.length.equals(numFeatures))


    val avgBias: Double = biasValues.map(f => { f(predictionLabelIndex)/f.sum }).sum/getNumTrees.toDouble
    val predictedLabelProbability = probabilities(predictionLabelIndex)

    val checkSum: Double = averagedContributions.sum + avgBias

    val checkSumToProbabilityRatio: Double = if (predictedLabelProbability > 0) {
      checkSum/predictedLabelProbability
    } else {
      0.0
    }

    if (scala.math.abs(checkSumToProbabilityRatio - 1) > .2) {
      println(s"kinda far out features: $features")
      println(s" -- predictionLabel: $prediction, predictedLabelProbability: $predictedLabelProbability, bias: $avgBias, checkSum: $checkSum")
      println(s" -- probabilities: ${probabilities.toArray.mkString(", ")}")
      println(s" -- averagedContributions: ${averagedContributions.mkString(", ")}")
      println(s" -- checkSumToProbabilityRatio: $checkSumToProbabilityRatio")
    } else {
      println(s"ok: $features")
      println(s" -- predictionLabel: $prediction, predictedLabelProbability: $predictedLabelProbability, bias: $avgBias, checkSum: $checkSum")
      println(s" -- probabilities: ${probabilities.toArray.mkString(", ")}")
      println(s" -- averagedContributions: ${averagedContributions.mkString(", ")}")
      println(s" -- checkSumToProbabilityRatio: $checkSumToProbabilityRatio")
    }

    //assert(scala.math.abs((checkSum/predictedLabelProbability)-1)<.2)

    val x: Array[Double] = Array(prediction, avgBias, checkSum) ++ probabilities.toArray ++ averagedContributions

    new org.apache.spark.ml.linalg.DenseVector(x)
  }

  override protected def raw2probability(rawPrediction: Vector): Vector = {
    val probs = rawPrediction.copy
    raw2probabilityInPlace(probs)
  }
}


