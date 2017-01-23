import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{InterpretedRandomForestClassificationModel, InterpretedRandomForestClassifier, RandomForestClassifier}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.tree.ImpactExtractor
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.scalatest.FunSuite

class TestInterpretedRandomForestClassifier extends FunSuite with TestSparkContext {
  val features = Array("sepal_length", "sepal_width", "petal_length", "petal_width")

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

    val testFeatures = new DenseVector(Array[Double](6.0,3.4,4.5,1.6))
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

  test("Verify code for impact JSON") {
    val modelInfo = interpretedModelPipeline._2
    val rfModel = modelInfo
    val testFeatures = Array[Double](5.5,2.6,4.4,1.2) // "91",5.5,2.6,4.4,1.2,"versicolor"

    val dv = new DenseVector(testFeatures)
    val v = rfModel.interpretedPrediction(dv).toArray.takeRight(testFeatures.length)
    val impactJson = ImpactExtractor.labeledContributionsAsJson(new DenseVector(v), features.mkString(","), dv)
    assert(impactJson.equals(expectedJSONString))
  }

  test("Verify impact JSON from UDF") {
    val predictions = interpretedModelPipeline._3.where("observation = 91") // "91",5.5,2.6,4.4,1.2,"versicolor"
    val withImpactJson = predictions.selectExpr("*", s"labeledContributionsAsJson(contributions, '${features.mkString(",")}', features) as impact_json")
    val impactRow = withImpactJson.select("impact_json").collect()(0)
    val impactJson = impactRow.getString(0)
    assert(impactJson.equals(expectedJSONString))
  }

  test("Check overall model field importances") {
    println("Interpreted-RF importances")
    var fIndex = 0
    val imp = interpretedModelPipeline._2.featureImportances.toArray.map(f => {
      val str = s"${features(fIndex)} - $f"
      fIndex += 1
      str
    }).mkString("\n")

    assert(imp.equals(expectedFieldImportances))
  }

  test("labels match standard random forest label output") {
    val predictions = interpretedModelPipeline._3
    val rfDF = rfPredictions().withColumnRenamed("observation", "rf_observation").withColumnRenamed("predictedLabel", "rf_predictedLabel")
    val j = predictions.join(rfDF, predictions("observation") === rfDF("rf_observation") && predictions("predictedLabel") === rfDF("rf_predictedLabel"))
    assert(predictions.count().equals(j.count()))
  }

  test("Pipeline can be saved") {
    interpretedModelPipeline._1.write.overwrite().save(resourcePath("irisInterpretedPipeline"))
  }

  test("Pipeline can be read") {
    val p = PipelineModel.load(resourcePath("irisInterpretedPipeline"))
    val predictions = p.transform(interpretedModelPipeline._4)
    println(predictions.schema.fieldNames)
    println(predictions.count())
  }

  def createInterpretedPipeline(): (PipelineModel, InterpretedRandomForestClassificationModel, DataFrame, DataFrame) = {
    val df = loadIrisData()
    val transformedValues = transformedDF(df, features)

    df.sparkSession.udf.register("labeledContributionsAsJson", (contributions: DenseVector, featureList: String, featureValues: DenseVector) =>
      ImpactExtractor.labeledContributionsAsJson(contributions, featureList, featureValues))

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

  private def loadIrisData(): DataFrame = {
    sqlContext.read
      .option("header", "true") // Use first line of all files as header
      .option("inferSchema", "true") // Automatically infer data types
      .csv(resourcePath("iris.csv"))
  }

  val expectedJSONString: String =
    """{
      |  "contributions": [
      |    {
      |      "feature_index": 0,
      |      "field_name": "sepal_length",
      |      "feature_impact": 0.010901162790697671,
      |      "feature_value": 5.5
      |    },
      |    {
      |      "feature_index": 1,
      |      "field_name": "sepal_width",
      |      "feature_impact": 0.0,
      |      "feature_value": 2.6
      |    },
      |    {
      |      "feature_index": 2,
      |      "field_name": "petal_length",
      |      "feature_impact": 0.2241226097580585,
      |      "feature_value": 4.4
      |    },
      |    {
      |      "feature_index": 3,
      |      "field_name": "petal_width",
      |      "feature_impact": 0.45881226522133056,
      |      "feature_value": 1.2
      |    }
      |  ]
      |}""".stripMargin

  val expectedFieldImportances: String =
    """sepal_length - 0.0232888015166167
      |sepal_width - 0.01262475537984371
      |petal_length - 0.3777193883591449
      |petal_width - 0.5863670547443947""".stripMargin
}
