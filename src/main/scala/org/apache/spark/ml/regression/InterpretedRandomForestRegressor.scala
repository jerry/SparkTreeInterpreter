package org.apache.spark.ml.regression

import org.apache.spark.annotation.Since
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.tree._
import org.apache.spark.ml.tree.impl.RandomForest
import org.apache.spark.ml.util.DefaultParamsReader.Metadata
import org.apache.spark.ml.util._
import org.apache.spark.mllib.tree.configuration.{Algo => OldAlgo}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.json4s.{DefaultFormats, JObject}
import org.json4s.JsonDSL._

class InterpretedRandomForestRegressor(override val uid: String) extends RandomForestRegressor {
  def this() = this(Identifiable.randomUID("rfr"))

  override protected def train(dataset: Dataset[_]): RandomForestRegressionModel = {
    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))
    val oldDataset: RDD[LabeledPoint] = extractLabeledPoints(dataset)
    val strategy =
      super.getOldStrategy(categoricalFeatures, numClasses = 0, OldAlgo.Regression, getOldImpurity)

    val instr = Instrumentation.create(this, oldDataset)
    instr.logParams(params: _*)

    val trees = RandomForest
      .run(oldDataset, strategy, getNumTrees, getFeatureSubsetStrategy, getSeed, Some(instr))
      .map(_.asInstanceOf[DecisionTreeRegressionModel])

    val numFeatures = oldDataset.first().features.size
    val m = new InterpretedRandomForestRegressionModel(trees, numFeatures)
    instr.logSuccess(m)
    m
  }
}

object InterpretedRandomForestRegressor extends RandomForestRegressor {
  /** Accessor for supported impurity settings: variance */
  final val supportedImpurities: Array[String] = TreeRegressorParams.supportedImpurities

  /** Accessor for supported featureSubsetStrategy settings: auto, all, onethird, sqrt, log2 */
  final val supportedFeatureSubsetStrategies: Array[String] = RandomForestParams.supportedFeatureSubsetStrategies

  //override def load(path: String): InterpretedRandomForestRegressor = super.load(path)
}

/**
  * [[http://en.wikipedia.org/wiki/Random_forest  Random Forest]] model for regression.
  * It supports both continuous and categorical features.
  *
  * @param _trees  Decision trees in the ensemble.
  * @param numFeatures  Number of features used by this model
  */
class InterpretedRandomForestRegressionModel private[ml] (
                                                override val uid: String,
                                                private val _trees: Array[DecisionTreeRegressionModel],
                                                override val numFeatures: Int)
  extends RandomForestRegressionModel (uid: String, _trees: Array[DecisionTreeRegressionModel], numFeatures: Int) {

  require(_trees.nonEmpty, "RandomForestRegressionModel requires at least 1 tree.")

  /**
    * Construct a random forest regression model, with all trees weighted equally.
    *
    * @param trees  Component trees
    */
  private[ml] def this(trees: Array[DecisionTreeRegressionModel], numFeatures: Int) =
    this(Identifiable.randomUID("rfr"), trees, numFeatures)

  override protected def transformImpl(dataset: Dataset[_]): DataFrame = {
    val bcastModel = dataset.sparkSession.sparkContext.broadcast(this)
    val predictUDF = udf { (features: Any) =>
      bcastModel.value.interpretedPrediction(features.asInstanceOf[Vector])
    }
    dataset.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
  }

  /**
    * Transforms dataset by reading from [[featuresCol]], and appending new columns as specified by
    * parameters:
    *  - predicted labels as [[predictionCol]] of type [[Double]]
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

    val extractBiasUDF = udf { (interpretedPrediction: Any) =>
      extractPredictionElement(interpretedPrediction.asInstanceOf[Vector], 1)
    }
    outputData = outputData.withColumn("bias", extractBiasUDF(col("interpretedPrediction")))

    val extractChecksumUDF = udf { (interpretedPrediction: Any) =>
      extractPredictionElement(interpretedPrediction.asInstanceOf[Vector], 2)
    }
    outputData = outputData.withColumn("checksum", extractChecksumUDF(col("interpretedPrediction")))

    val extractContributionsUDF = udf { (interpretedPrediction: Any) =>
      extractContributions(interpretedPrediction.asInstanceOf[Vector])
    }
    outputData = outputData.withColumn("contributions", extractContributionsUDF(col("interpretedPrediction")))

    outputData.toDF
  }

  def extractPredictionElement(interpretedPrediction: Vector, elementId: Int): Double = {
    interpretedPrediction(elementId)
  }

  def extractContributions(interpretedPrediction: Vector): Vector = {
    Vectors.dense(interpretedPrediction.toArray.view(3, numFeatures + 3).toArray)
  }

  /** Get a prediction, with bias, checksum, and, per-feature contribution amounts
    *
    * @param features vector of feature values
    * @return prediction, bias, checksum, contribution amounts
    */
  def interpretedPrediction(features: Vector): Vector = {
    var sumOfPredictions: Double = 0.0
    var sumOfBias: Double = 0.0

    // per tree, per prediction, the impact of each internal node on the final prediction
    val treeContributions = new Array[Option[Array[FeatureContribution]]](getNumTrees)
    var nextTree: Int = 0

    _trees.view.foreach { tree =>
      val rootNode = tree.rootNode
      val transparentLeafNode: TransparentNode = new TransparentNode(rootNode, 0, features, rootNode = true).interpretationImpl()

      sumOfPredictions += transparentLeafNode.prediction
      sumOfBias += rootNode.prediction

      treeContributions.update(nextTree, transparentLeafNode.featureContributions())
      nextTree += 1
    }

    val prediction = sumOfPredictions / getNumTrees
    val avgBias: Double = sumOfBias / getNumTrees

    val allAveragedContributions: Array[Array[Double]] = averageContributions(treeContributions, features) // Get the contributions for each possible label
    assert(allAveragedContributions.length.equals(numFeatures))

    // Use the contributions from the actual label
    val averagedContributions: Array[Double] = allAveragedContributions.map(b => b(0))
    assert(averagedContributions.length.equals(numFeatures))

    val checkSum: Double = averagedContributions.sum + avgBias
    val x: Array[Double] = Array(prediction, avgBias, checkSum) ++ averagedContributions

    Vectors.dense(x)
  }

  /** Contributions are the contributions to a given prediction by the ensemble of decision trees, where each represents the change in probability of the
    * predicted label at each step in each decision tree.
    *
    * comes in as 1 per class, per feature, per tree
    * leaves as an array of doubles (1 per class), per feature
    *
    */
  def averageContributions(contributions: Array[Option[Array[FeatureContribution]]], features: Vector): Array[Array[Double]] = {
    val allContributions: Array[FeatureContribution] = contributions.flatMap(f => f.getOrElse(new Array[FeatureContribution](1)))
    val avgContributions: Array[Array[Double]] = (0 until features.size).toArray.map(f => {
      // get the per-class contributions for the current featureIndex, f
      val i: Array[FeatureContribution] = allContributions.filter(p => p.featureIndex.equals(f))
      val contributionCount = getNumTrees

      Array(0).map(j => {
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


  override def toString: String = {
    s"InterpretedRandomForestRegressionModel (uid=$uid) with $getNumTrees trees"
  }
}

object InterpretedRandomForestRegressionModel extends MLReadable[InterpretedRandomForestRegressionModel] {

  @Since("2.0.0")
  override def read: MLReader[InterpretedRandomForestRegressionModel] = new InterpretedRandomForestRegressionModelReader

  @Since("2.0.0")
  override def load(path: String): InterpretedRandomForestRegressionModel = super.load(path)

  private[InterpretedRandomForestRegressionModel]
  class InterpretedRandomForestRegressionModelWriter(instance: InterpretedRandomForestRegressionModel)
    extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      val extraMetadata: JObject = Map(
        "numFeatures" -> instance.numFeatures,
        "numTrees" -> instance.getNumTrees)
      EnsembleModelReadWrite.saveImpl(instance, path, sparkSession, extraMetadata)
    }
  }

  private class InterpretedRandomForestRegressionModelReader extends MLReader[InterpretedRandomForestRegressionModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[InterpretedRandomForestRegressionModel].getName
    private val treeClassName = classOf[DecisionTreeRegressionModel].getName

    override def load(path: String): InterpretedRandomForestRegressionModel = {
      implicit val format = DefaultFormats
      val (metadata: Metadata, treesData: Array[(Metadata, Node)], treeWeights: Array[Double]) =
        EnsembleModelReadWrite.loadImpl(path, sparkSession, className, treeClassName)
      val numFeatures = (metadata.metadata \ "numFeatures").extract[Int]
      val numTrees = (metadata.metadata \ "numTrees").extract[Int]

      val trees: Array[DecisionTreeRegressionModel] = treesData.map { case (treeMetadata, root) =>
        val tree =
          new DecisionTreeRegressionModel(treeMetadata.uid, root, numFeatures)
        DefaultParamsReader.getAndSetParams(tree, treeMetadata)
        tree
      }
      require(numTrees == trees.length, s"InterpretedRandomForestRegressionModel.load expected $numTrees" +
        s" trees based on metadata but found ${trees.length} trees.")

      val model = new InterpretedRandomForestRegressionModel(metadata.uid, trees, numFeatures)
      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }
}
