import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.tree.{InterpretedRandomForestClassificationModel, InterpretedRandomForestClassifier}
import org.scalatest.FunSuite

class DressedForestTest extends FunSuite with TestSparkContext {
  test("Interpreted Random Forest Classification Test") {
    import spark.implicits._

    val modelInfo = createInterpretedModel()
    val rfModel = modelInfo

    val testFeatures = Array[Double](6.0,3.4,4.5,1.6)
    val v = rfModel.interpretedPrediction(testFeatures)
    assert(v.size.equals(10))
    println(v.toString)

//    val treeData = sqlContext.read.load(s"$modelPath/treesMetadata").as[ShadowTree]
//    val nodeData = sqlContext.read.load(s"$modelPath/data").as[ShadowTreeNode]
//
//    val nodeCount = nodeData.count()
//    val sf = ShadowForest(10, treeData, nodeData, rfModel.numClasses)
//    val treeNodeCount = sf.treeMap.values.map(_.nodeCount()).sum
//
//    assert(treeNodeCount == nodeCount)
    // val interpretedPredictions: DataFrame = sf.transform(testData)
  }

  test("Random Forest Classification Test") {
//    import spark.implicits._
//
//    val modelInfo = loadOrCreateModel()
//    val rfModel = modelInfo._1
//    val modelPath = modelInfo._2
//
//    val treeData = sqlContext.read.load(s"$modelPath/treesMetadata").as[ShadowTree]
//    val nodeData = sqlContext.read.load(s"$modelPath/data").as[ShadowTreeNode]
//
//    val nodeCount = nodeData.count()
//    val sf = ShadowForest(10, treeData, nodeData, rfModel.numClasses)
//    val treeNodeCount = sf.treeMap.values.map(_.nodeCount()).sum
//
//    assert(treeNodeCount == nodeCount)
    // val interpretedPredictions: DataFrame = sf.transform(testData)
  }

  def loadOrCreateModel(): (RandomForestClassificationModel, String) = {
    val currentDir = System.getProperty("user.dir")
    val resourcesPath = s"$currentDir/src/test/resources"
    val replacementPath = s"$resourcesPath/iris.csv"
    val modelPath = s"$resourcesPath/irisRandomForestClassificationModel"

    val conf = sc.hadoopConfiguration
    val fs = org.apache.hadoop.fs.FileSystem.get(conf)

    val rfModelObject = if (!fs.exists(new org.apache.hadoop.fs.Path(modelPath))) {
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

      val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
      println("Learned classification forest model:\n" + rfModel.toDebugString)
      rfModel.save(modelPath)
      rfModel
    } else {
      RandomForestClassificationModel.load(modelPath)
    }
    (rfModelObject, modelPath)
  }

  def createInterpretedModel(): InterpretedRandomForestClassificationModel = {
    val currentDir = System.getProperty("user.dir")
    val resourcesPath = s"$currentDir/src/test/resources"
    val replacementPath = s"$resourcesPath/iris.csv"

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

    val Array(trainingData, testData) = transformedValues.randomSplit(Array(0.7, 0.3), 306090)

    val rf = new InterpretedRandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(10)
      .setSeed(90210)

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))
    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf))
    val model = pipeline.fit(trainingData)

    // predictions.select("predictedLabel", "species", "indexedLabel", "features").show(50)
    val rfModel = model.stages(2).asInstanceOf[InterpretedRandomForestClassificationModel]

    var fCount = 0
    rfModel.featureImportances.toArray.foreach(f => {
      println(s"$f - ${features(fCount)}")
      fCount += 1
    })

    val predictions = model.transform(testData)
    predictions.show(5000)

    // println("Learned classification forest model:\n" + rfModel.toDebugString)
    rfModel
  }
}














