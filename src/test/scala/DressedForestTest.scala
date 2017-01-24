import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.tree.{InterpretedRandomForestClassificationModel, InterpretedRandomForestClassifier}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.scalatest.FunSuite

class DressedForestTest extends FunSuite with TestSparkContext {
  private var _interpretedModelPipeline: Option[(PipelineModel, InterpretedRandomForestClassificationModel, Dataset[Row], Dataset[Row])] = None

  lazy val interpretedModelPipeline: (PipelineModel, InterpretedRandomForestClassificationModel, Dataset[Row], Dataset[Row]) = {
    _interpretedModelPipeline.getOrElse({
      val pl = createInterpretedPipeline()
      _interpretedModelPipeline = Some(pl)
      pl
    })
  }

  test("Interpreted Random Forest Classification Model Test") {
    val modelInfo = interpretedModelPipeline._2
    val rfModel = modelInfo

    val testFeatures = Array[Double](6.0,3.4,4.5,1.6)
    val v = rfModel.interpretedPrediction(testFeatures)

    assert(v.size.equals(11)) // we have all our fields
    assert(v(0).equals(0.0)) // It is a versicolor.
    assert(v(1).equals(1.0)) // We are sure about it.
    assert(v(7) > 0.0) // importance of sepal_length is > 0
    assert(v(8) > 0.0) // importance of sepal_width is > 0
    assert(v(9) > 0.0) // importance of petal_length is > 0
    assert(v(10) > 0.0) // importance of petal_width is > 0
  }

  test("Interpreted Random Forest Classification Transform Test") {
    val predictions = interpretedModelPipeline._3
    val withSpec = predictions.selectExpr("*", "abs(checksum - predictedLabelProbability) as checksumVariance")
    val outOfSpec = withSpec.where("checksumVariance > 0.1")
    val outOfSpecCount = outOfSpec.count()
    if (outOfSpecCount > 0) {
      outOfSpec.show(outOfSpecCount.toInt)
    }
    assert(outOfSpecCount == 0)
  }

  test("Dare to Compare Labels") {
    val predictions = interpretedModelPipeline._3
    val rfDF = rfPredictions().withColumnRenamed("observation", "rf_observation").withColumnRenamed("predictedLabel", "rf_predictedLabel")
    val j = predictions.join(rfDF, predictions("observation") === rfDF("rf_observation") && predictions("predictedLabel") === rfDF("rf_predictedLabel"))
    assert(predictions.count().equals(j.count()))
  }

  test("Can we save?") {
    interpretedModelPipeline._1.write.overwrite().save(resourcePath("irisInterpretedPipeline"))
  }

  test("Can we read?") {
    val p = PipelineModel.load(resourcePath("irisInterpretedPipeline"))
    val predictions = p.transform(interpretedModelPipeline._4)
    println(predictions.schema.fieldNames)
    println(predictions.count())
  }

  def createInterpretedPipeline(): (PipelineModel, InterpretedRandomForestClassificationModel, DataFrame, DataFrame) = {
    val df = loadIrisData()
    val features = Array("sepal_length", "sepal_width", "petal_length", "petal_width")
    val transformedValues = transformedDF(df, features)

    val labelIndexer = new StringIndexer().setInputCol("species").setOutputCol("indexedLabel").fit(transformedValues)
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(transformedValues)

    val Array(trainingData, testData) = transformedValues.randomSplit(Array(0.7, 0.3), 306090)

    val irf = new InterpretedRandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(10)
      .setSeed(90210)

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, irf, labelConverter))
    val model = pipeline.fit(trainingData)
    val rfModel = model.stages(2).asInstanceOf[InterpretedRandomForestClassificationModel]

//    var fCount = 0
//    println("Interpreted-RF importances")
//    rfModel.featureImportances.toArray.foreach(f => {
//      println(s"$f - ${features(fCount)}")
//      fCount += 1
//    })

    (model, rfModel, model.transform(testData), testData)
  }


  def rfPredictions(): DataFrame = {
    val df = loadIrisData()
    val features = Array("sepal_length", "sepal_width", "petal_length", "petal_width")
    val transformedValues = transformedDF(df, features)

    val labelIndexer = new StringIndexer().setInputCol("species").setOutputCol("indexedLabel").fit(transformedValues)
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(transformedValues)

    val Array(trainingData, testData) = transformedValues.randomSplit(Array(0.7, 0.3), 306090)

    val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(10)
      .setSeed(90210)

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))
    val model = pipeline.fit(trainingData)
    model.transform(testData)
  }

  override def resourcePath(fileOrDirectory: String): String = {
    val currentDir = System.getProperty("user.dir")
    val resourcesPath = s"$currentDir/src/test/resources"
    s"$resourcesPath/$fileOrDirectory"
  }

  private def loadIrisData(): DataFrame = {
    sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true") // Use first line of all files as header
      .option("inferSchema", "true") // Automatically infer data types
      .load(resourcePath("iris.csv"))
  }

  override def transformedDF(df: DataFrame, features: Array[String]): DataFrame = {
    new VectorAssembler().setInputCols(features).setOutputCol("features").transform(df)
  }
}