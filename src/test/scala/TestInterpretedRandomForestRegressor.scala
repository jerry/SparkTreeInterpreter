import org.apache.spark.ml.interpretedrandomforest.regression.{InterpretedRandomForestRegressionModel, InterpretedRandomForestRegressor}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.tree.ImpactExtractor
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.scalatest.FunSuite

class TestInterpretedRandomForestRegressor extends FunSuite with TestSparkContext {
  val features: Array[String] = Array("crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "b", "lstat")

  private var _interpretedModelPipeline: Option[(PipelineModel, InterpretedRandomForestRegressionModel, Dataset[Row], Dataset[Row])] = None

  lazy val interpretedModelPipeline: (PipelineModel, InterpretedRandomForestRegressionModel, Dataset[Row], Dataset[Row]) = {
    _interpretedModelPipeline.getOrElse({
      val pl = createInterpretedPipeline()
      _interpretedModelPipeline = Some(pl)
      pl
    })
  }

  test("Interpreted Random Forest Regression Model Test") {
    val modelInfo = interpretedModelPipeline._2
    val rfModel = modelInfo

    val testFeatures = new DenseVector(Array[Double](0.22489,12.5,7.87,0,0.524,6.377,94.3,6.3467,5,311,15.2,392.52,20.45))
    val v = rfModel.interpretedPrediction(testFeatures)

    assert(v.size.equals(16)) // we have all our fields
    assert(v(0).equals(18.456321428571428)) // predicted value
    assert(v(1).equals(22.23943523616007)) // bias
    assert(v(2).equals(18.456321428571428)) // checksum
  }

  test("Interpreted Random Forest Regression Model RMSE") {
    val predictions = interpretedModelPipeline._3
    val evaluator = new RegressionEvaluator()
      .setLabelCol("medv")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println("Root Mean Squared Error (RMSE) on test data for Interpreted Random Forest Regression Model = " + rmse)
    assert(rmse.equals(5.197421959850002))
  }

  test("Verify code for impact JSON") {
    val modelInfo = interpretedModelPipeline._2
    val rfModel = modelInfo

    val testFeatures = new DenseVector(Array[Double](0.22489,12.5,7.87,0,0.524,6.377,94.3,6.3467,5,311,15.2,392.52,20.45))
    val v = rfModel.interpretedPrediction(testFeatures).toArray.takeRight(testFeatures.size)

    val impactJson = ImpactExtractor.labeledContributionsAsJson(new DenseVector(v), features.mkString(","), testFeatures)
    assert(impactJson.equals(expectedJSONString))
  }

  test("Verify impact JSON from UDF") {
    val predictions = interpretedModelPipeline._3.where("crim = 0.22489")
    val withImpactJson = predictions.selectExpr("*", s"labeledContributionsAsJson(contributions, '${features.mkString(",")}', features) as impact_json")
    val impactRow = withImpactJson.select("impact_json").collect()(0)
    val impactJson = impactRow.getString(0)
    assert(impactJson.equals(expectedJSONString))
  }

  test("Interpreted Random Forest Regression Transform Test") {
    val predictions = interpretedModelPipeline._3
    val withSpec = predictions.selectExpr("*", "abs(checksum - prediction) as checksumVariance")
    val outOfSpec = withSpec.where("checksumVariance > 0.01")
    val outOfSpecCount = outOfSpec.count()
    if (outOfSpecCount > 0) {
      outOfSpec.show(outOfSpecCount.toInt)
    }
    assert(outOfSpecCount == 0)
  }

  test("labels match standard random forest label output") {
    val predictions = interpretedModelPipeline._3
    val rfDF = rfPredictions().withColumnRenamed("crim", "rf_crim").withColumnRenamed("zn", "rf_zn").withColumnRenamed("prediction", "rf_prediction")
    val j = predictions.join(rfDF, predictions("crim") === rfDF("rf_crim") && predictions("zn") === rfDF("rf_zn") &&
      predictions("prediction") === rfDF("rf_prediction"))
    assert(predictions.count().equals(j.count()))
  }

  test("Pipeline can be saved") {
    interpretedModelPipeline._1.write.overwrite().save(resourcePath("bostonInterpretedPipeline"))
  }

  test("Pipeline can be read") {
    val p = PipelineModel.load(resourcePath("bostonInterpretedPipeline"))
    val predictions = p.transform(interpretedModelPipeline._4)
    println(predictions.schema.fieldNames)
    println(predictions.count())
  }

  def createInterpretedPipeline(): (PipelineModel, InterpretedRandomForestRegressionModel, DataFrame, DataFrame) = {
    val df = loadBostonData()

    val features = Array("crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "b", "lstat")
    val transformedValues = transformedDF(df, features)
    val Array(trainingData, testData) = transformedValues.randomSplit(Array(0.2, 0.8), seed = 5)
    println(transformedValues.schema.printTreeString())

    df.sparkSession.udf.register("labeledContributionsAsJson", (contributions: DenseVector, featureList: String, featureValues: DenseVector) =>
      ImpactExtractor.labeledContributionsAsJson(contributions, featureList, featureValues))

    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(transformedValues)

    val rf = new InterpretedRandomForestRegressor()
      .setLabelCol("medv")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(10)
      .setSeed(90210)

    val pipeline = new Pipeline().setStages(Array(featureIndexer, rf))

    println(trainingData.schema.printTreeString())
    val model = pipeline.fit(trainingData)
    val rfModel = model.stages(1).asInstanceOf[InterpretedRandomForestRegressionModel]

    (model, rfModel, model.transform(testData), testData)
  }

  def rfPredictions(): DataFrame = {
    val df = loadBostonData()
    val transformedValues = transformedDF(df, features)
    val Array(trainingData, testData) = transformedValues.randomSplit(Array(0.2, 0.8), seed = 5)
    println(transformedValues.schema.printTreeString())

    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(transformedValues)

    val rf = new RandomForestRegressor()
      .setLabelCol("medv")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(10)
      .setSeed(90210)

    val pipeline = new Pipeline().setStages(Array(featureIndexer, rf))

    println(trainingData.schema.printTreeString())
    val model = pipeline.fit(trainingData)
    val output = model.transform(testData)

    val evaluator = new RegressionEvaluator()
      .setLabelCol("medv")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(output)

    println("Root Mean Squared Error (RMSE) on test data for RandomForestRegressor = " + rmse)
    assert(rmse.equals(5.197421959850002))

    println(output.schema.printTreeString())
    println(output.show(50))
    output
  }

  private def loadBostonData(): DataFrame = {
    sqlContext.read
      .option("header", "true") // Use first line of all files as header
      .option("inferSchema", "true") // Automatically infer data types
      .csv(resourcePath("bostonData.data"))
  }

  val expectedJSONString: String =
    """{
      |  "contributions": [
      |    {
      |      "feature_index": 0,
      |      "field_name": "crim",
      |      "feature_impact": 1.343690587552849,
      |      "feature_value": 0.22489
      |    },
      |    {
      |      "feature_index": 1,
      |      "field_name": "zn",
      |      "feature_impact": 0.9604070660522247,
      |      "feature_value": 12.5
      |    },
      |    {
      |      "feature_index": 2,
      |      "field_name": "indus",
      |      "feature_impact": -1.5161285760518008,
      |      "feature_value": 7.87
      |    },
      |    {
      |      "feature_index": 3,
      |      "field_name": "chas",
      |      "feature_impact": 0.0,
      |      "feature_value": 0.0
      |    },
      |    {
      |      "feature_index": 4,
      |      "field_name": "nox",
      |      "feature_impact": 0.6627663861521629,
      |      "feature_value": 0.524
      |    },
      |    {
      |      "feature_index": 5,
      |      "field_name": "rm",
      |      "feature_impact": -1.672368988625827,
      |      "feature_value": 6.377
      |    },
      |    {
      |      "feature_index": 6,
      |      "field_name": "age",
      |      "feature_impact": -2.067060669645857,
      |      "feature_value": 94.3
      |    },
      |    {
      |      "feature_index": 7,
      |      "field_name": "dis",
      |      "feature_impact": 0.7189812540400778,
      |      "feature_value": 6.3467
      |    },
      |    {
      |      "feature_index": 8,
      |      "field_name": "rad",
      |      "feature_impact": 0.0,
      |      "feature_value": 5.0
      |    },
      |    {
      |      "feature_index": 9,
      |      "field_name": "tax",
      |      "feature_impact": -0.8896904112887926,
      |      "feature_value": 311.0
      |    },
      |    {
      |      "feature_index": 10,
      |      "field_name": "ptratio",
      |      "feature_impact": 1.2904412877857692,
      |      "feature_value": 15.2
      |    },
      |    {
      |      "feature_index": 11,
      |      "field_name": "b",
      |      "feature_impact": 0.2462682072829125,
      |      "feature_value": 392.52
      |    },
      |    {
      |      "feature_index": 12,
      |      "field_name": "lstat",
      |      "feature_impact": -2.8604199508423616,
      |      "feature_value": 20.45
      |    }
      |  ]
      |}""".stripMargin
}


