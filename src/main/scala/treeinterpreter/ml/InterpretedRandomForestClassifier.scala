package treeinterpreter.ml

import org.json4s.{DefaultFormats, JObject}
import org.json4s.JsonDSL._
import org.apache.spark.annotation.Since
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, ProbabilisticClassificationModel, ProbabilisticClassifier}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tree.{RandomForestParams, _}
import org.apache.spark.ml.tree.impl.RandomForest
import org.apache.spark.ml.util._
import org.apache.spark.ml.util.DefaultParamsReader.Metadata
import org.apache.spark.mllib.tree.configuration.{Algo => OldAlgo}
import org.apache.spark.mllib.tree.model.{RandomForestModel => OldRandomForestModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions._


/**
  * [[http://en.wikipedia.org/wiki/Random_forest  Random Forest]] learning algorithm for
  * classification.
  * It supports both binary and multiclass labels, as well as both continuous and categorical
  * features.
  */

class InterpretedRandomForestClassifier  (override val uid: String)
  extends ProbabilisticClassifier[Vector, InterpretedRandomForestClassifier, InterpretedRandomForestClassificationModel]
    with InterpretedRandomForestClassifierParams with DefaultParamsWritable {

  
  def this() = this(Identifiable.randomUID("rfc"))

  // Override parameter setters from parent trait for Java API compatibility.

  // Parameters from TreeClassifierParams:

  
  override def setMaxDepth(value: Int): this.type = super.setMaxDepth(value)

  
  override def setMaxBins(value: Int): this.type = super.setMaxBins(value)

  
  override def setMinInstancesPerNode(value: Int): this.type =
    super.setMinInstancesPerNode(value)

  
  override def setMinInfoGain(value: Double): this.type = super.setMinInfoGain(value)

  
  override def setMaxMemoryInMB(value: Int): this.type = super.setMaxMemoryInMB(value)

  
  override def setCacheNodeIds(value: Boolean): this.type = super.setCacheNodeIds(value)

  
  override def setCheckpointInterval(value: Int): this.type = super.setCheckpointInterval(value)

  
  override def setImpurity(value: String): this.type = super.setImpurity(value)

  // Parameters from TreeEnsembleParams:

  
  override def setSubsamplingRate(value: Double): this.type = super.setSubsamplingRate(value)

  
  override def setSeed(value: Long): this.type = super.setSeed(value)

  // Parameters from RandomForestParams:

  
  override def setNumTrees(value: Int): this.type = super.setNumTrees(value)

  
  override def setFeatureSubsetStrategy(value: String): this.type =
    super.setFeatureSubsetStrategy(value)

  override protected def train(dataset: Dataset[_]): InterpretedRandomForestClassificationModel = {
    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))
    val numClasses: Int = getNumClasses(dataset)

    if (isDefined(thresholds)) {
      require($(thresholds).length == numClasses, this.getClass.getSimpleName +
        ".train() called with non-matching numClasses and thresholds.length." +
        s" numClasses=$numClasses, but thresholds has length ${$(thresholds).length}")
    }

    val oldDataset: RDD[LabeledPoint] = extractLabeledPoints(dataset, numClasses)
    val strategy =
      super.getOldStrategy(categoricalFeatures, numClasses, OldAlgo.Classification, getOldImpurity)

    val instr = Instrumentation.create(this, oldDataset)
    instr.logParams(params: _*)

    val trees = RandomForest
      .run(oldDataset, strategy, getNumTrees, getFeatureSubsetStrategy, getSeed, Some(instr))
      .map(_.asInstanceOf[DecisionTreeClassificationModel])

    val numFeatures = oldDataset.first().features.size
    val m = new RandomForestClassificationModel(trees, numFeatures, numClasses)
    instr.logSuInterpretedccess(m)
    m
  }

  @Since("1.4.1")
  override def copy(extra: ParamMap): InterpretedRandomForestClassifier = defaultCopy(extra)
}


object InterpretedRandomForestClassifier extends DefaultParamsReadable[InterpretedRandomForestClassifier] {
  /** Accessor for supported impurity settings: entropy, gini */
  
  final val supportedImpurities: Array[String] = TreeClassifierParams.supportedImpurities

  /** Accessor for supported featureSubsetStrategy settings: auto, all, onethird, sqrt, log2 */
  
  final val supportedFeatureSubsetStrategies: Array[String] =
  RandomForestParams.supportedFeatureSubsetStrategies

  @Since("2.0.0")
  override def load(path: String): InterpretedRandomForestClassifier = super.load(path)
}

/**
  * [[http://en.wikipedia.org/wiki/Random_forest  Random Forest]] model for classification.
  * It supports both binary and multiclass labels, as well as both continuous and categorical
  * features.
  *
  * @param _trees  Decision trees in the ensemble.
  *                Warning: These have null parents.
  */

class InterpretedRandomForestClassificationModel private[ml] (
                                                    @Since("1.5.0") override val uid: String,
                                                    private val _trees: Array[DecisionTreeClassificationModel],
                                                    @Since("1.6.0") override val numFeatures: Int,
                                                    @Since("1.5.0") override val numClasses: Int)
  extends ProbabilisticClassificationModel[Vector, InterpretedRandomForestClassificationModel]
    with InterpretedRandomForestClassificationModelParams with TreeEnsembleModel[DecisionTreeClassificationModel]
    with MLWritable with Serializable {

  require(_trees.nonEmpty, "RandomForestClassificationModel requires at least 1 tree.")

  /**
    * Construct a random forest classification model, with all trees weighted equally.
    *
    * @param trees  Component trees
    */
  private[ml] def this(
                        trees: Array[DecisionTreeClassificationModel],
                        numFeatures: Int,
                        numClasses: Int) =
    this(Identifiable.randomUID("rfc"), trees, numFeatures, numClasses)

  
  override def trees: Array[DecisionTreeClassificationModel] = _trees

  // Note: We may add support for weights (based on tree performance) later on.
  private lazy val _treeWeights: Array[Double] = Array.fill[Double](_trees.length)(1.0)

  
  override def treeWeights: Array[Double] = _treeWeights

  override protected def transformImpl(dataset: Dataset[_]): DataFrame = {
    val bcastModel = dataset.sparkSession.sparkContext.broadcast(this)
    val predictUDF = udf { (features: Any) =>
      bcastModel.value.predict(features.asInstanceOf[Vector])
    }
    dataset.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
  }

  override protected def predictRaw(features: Vector): Vector = {
    // TODO: When we add a generic Bagging class, handle transform there: SPARK-7128
    // Classifies using majority votes.
    // Ignore the tree weights since all are 1.0 for now.
    val votes = Array.fill[Double](numClasses)(0.0)
    _trees.view.foreach { tree =>
      val classCounts: Array[Double] = tree.rootNode.predictImpl(features).impurityStats.stats
      val total = classCounts.sum
      if (total != 0) {
        var i = 0
        while (i < numClasses) {
          votes(i) += classCounts(i) / total
          i += 1
        }
      }
    }
    Vectors.dense(votes)
  }

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction match {
      case dv: DenseVector =>
        ProbabilisticClassificationModel.normalizeToProbabilitiesInPlace(dv)
        dv
      case sv: SparseVector =>
        throw new RuntimeException("Unexpected error in RandomForestClassificationModel:" +
          " raw2probabilityInPlace encountered SparseVector")
    }
  }

  /**
    * Number of trees in ensemble
    *
    * @deprecated  Use [[getNumTrees]] instead.  This method will be removed in 2.1.0
    */
  // TODO: Once this is removed, then this class can inherit from RandomForestClassifierParams
  @deprecated("Use getNumTrees instead.  This method will be removed in 2.1.0.", "2.0.0")
  val numTrees: Int = trees.length

  
  override def copy(extra: ParamMap): InterpretedRandomForestClassificationModel = {
    copyValues(new InterpretedRandomForestClassificationModel(uid, _trees, numFeatures, numClasses), extra)
      .setParent(parent)
  }

  
  override def toString: String = {
    s"RandomForestClassificationModel (uid=$uid) with $getNumTrees trees"
  }

  /**
    * Estimate of the importance of each feature.
    *
    * Each feature's importance is the average of its importance across all trees in the ensemble
    * The importance vector is normalized to sum to 1. This method is suggested by Hastie et al.
    * (Hastie, Tibshirani, Friedman. "The Elements of Statistical Learning, 2nd Edition." 2001.)
    * and follows the implementation from scikit-learn.
    *
    * @see [[DecisionTreeClassificationModel.featureImportances]]
    */
  lazy val featureImportances: Vector = TreeEnsembleModel.featureImportances(trees, numFeatures)

  /** (private[ml]) Convert to a model in the old API */
  private[ml] def toOld: OldRandomForestModel = {
    new OldRandomForestModel(OldAlgo.Classification, _trees.map(_.toOld))
  }

  override def write: MLWriter =
    new InterpretedRandomForestClassificationModel.RandomForestClassificationModelWriter(this)
}

object InterpretedRandomForestClassificationModel extends MLReadable[InterpretedRandomForestClassificationModel] {

  override def read: MLReader[InterpretedRandomForestClassificationModel] =
    new InterpretedRandomForestClassificationModelReader

  override def load(path: String): InterpretedRandomForestClassificationModel = super.load(path)

  private[RandomForestClassificationModel]
  class RandomForestClassificationModelWriter(instance: InterpretedRandomForestClassificationModel)
    extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      // Note: numTrees is not currently used, but could be nice to store for fast querying.
      val extraMetadata: JObject = Map(
        "numFeatures" -> instance.numFeatures,
        "numClasses" -> instance.numClasses,
        "numTrees" -> instance.getNumTrees)
      EnsembleModelReadWrite.saveImpl(instance, path, sparkSession, extraMetadata)
    }
  }

  private class InterpretedRandomForestClassificationModelReader
    extends MLReader[InterpretedRandomForestClassificationModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[InterpretedRandomForestClassificationModel].getName
    private val treeClassName = classOf[DecisionTreeClassificationModel].getName

    override def load(path: String): InterpretedRandomForestClassificationModel = {
      implicit val format = DefaultFormats
      val (metadata: Metadata, treesData: Array[(Metadata, Node)], _) =
        EnsembleModelReadWrite.loadImpl(path, sparkSession, className, treeClassName)
      val numFeatures = (metadata.metadata \ "numFeatures").extract[Int]
      val numClasses = (metadata.metadata \ "numClasses").extract[Int]
      val numTrees = (metadata.metadata \ "numTrees").extract[Int]

      val trees: Array[DecisionTreeClassificationModel] = treesData.map {
        case (treeMetadata, root) =>
          val tree =
            new DecisionTreeClassificationModel(treeMetadata.uid, root, numFeatures, numClasses)
          DefaultParamsReader.getAndSetParams(tree, treeMetadata)
          tree
      }
      require(numTrees == trees.length, s"RandomForestClassificationModel.load expected $numTrees" +
        s" trees based on metadata but found ${trees.length} trees.")

      val model = new InterpretedRandomForestClassificationModel(metadata.uid, trees, numFeatures, numClasses)
      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }
}
