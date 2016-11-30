import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.sql.SQLContext
import org.scalatest.FunSuite
import treeinterpreter.{DressedClassifierForest, DressedClassifierTree}

class DressedForestTest extends FunSuite with SharedSparkContext {
  test("Random Forest Classification Test") {
    implicit val _sc = sc

    val sqlContext = new SQLContext(sc)

    val currentDir = System.getProperty("user.dir")
    val replacementPath = s"$currentDir/src/test/resources/iris.csv"

    val df = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true") // Use first line of all files as header
      .option("inferSchema", "true") // Automatically infer data types
      .load(replacementPath)

    val features = Array("sepal_length", "sepal_width", "petal_length", "petal_width")
    val vectorAssembler = new VectorAssembler().setInputCols(features).setOutputCol("features")
    val transformedValues = vectorAssembler.transform(df)

    val labelIndexer = new StringIndexer().setInputCol("species").setOutputCol("indexedLabel").fit(transformedValues)
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(transformedValues)

    val Array(trainingData, testData) = transformedValues.randomSplit(Array(0.7, 0.3))

    val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(10)

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))
    val model = pipeline.fit(trainingData)
    val predictions = model.transform(testData)
    predictions.select("predictedLabel", "species", "indexedLabel", "features").show(50)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("precision")

    val precision = evaluator.evaluate(predictions)
    println("Test Precision = " + precision)

    val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
    println("Learned classification forest model:\n" + rfModel.toDebugString)

    val t = new DressedClassifierForest(rfModel)
    t.interpret(testData)

//    val parsedData = data.map(line => {
//      val parsedLine = line.split(',')
//      LabeledPoint(parsedLine.last.toDouble, Vectors.dense(parsedLine.dropRight(1).map(_.toDouble)))
//    })
//
//    val splits = parsedData.randomSplit(Array(0.2, 0.8), seed = 5)
//    val (trainingData, testData) = (splits(0), splits(1))
//
//    val numClasses = 2
//    val categoricalFeaturesInfo = Map[Int, Int]()
//    val numTrees = 10
//    val featureSubsetStrategy = "auto"
//    val impurity = "variance"
//    val maxDepth = 2
//    val maxBins = 32
//
//    val classimpurity = "gini"
//
//    val rf: RandomForestModel = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo,
//      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, seed = 21)
//
//    val labelsAndPredictions = trainingData.map { point =>
//      val prediction = rf.predict(point.features)
//      (point.label, prediction)
//    }
//
//    val testMSE = math.sqrt(labelsAndPredictions.map { case (v, p) => math.pow(v - p, 2) }.mean())
//
//    println("Test Mean Squared Error = " + testMSE)
//
//    val interpRDD = Interp.interpretModel(rf, trainingData)
//
//    interpRDD.collect().foreach(println)
//
//    interpRDD.collect().foreach(item=> assert(scala.math.abs(item.checksum/item.prediction-1)<.2))
  }
}


