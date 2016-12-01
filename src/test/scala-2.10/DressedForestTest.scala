import org.apache.spark.SparkException
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{ProbabilisticClassificationModel, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.scalatest.FunSuite
import treeinterpreter.shadowforest.{LeafNode, ShadowForest, ShadowTree, ShadowTreeNode}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

class DressedForestTest extends FunSuite with TestSparkContext {
  test("Random Forest Classification Test") {
    implicit val _sc = sc

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
    // rfModel.save(s"$resourcesPath/irisRandomForestClassificationModel")

    import spark.implicits._

    val treeData = sqlContext.read.load(s"$resourcesPath/irisRandomForestClassificationModel/treesMetadata").as[ShadowTree]
    val nodeData = sqlContext.read.load(s"$resourcesPath/irisRandomForestClassificationModel/data").as[ShadowTreeNode]

    val nodeCount = nodeData.count()
    val sf = ShadowForest(10, treeData, nodeData, rfModel.numClasses)
    val treeNodeCount = sf.treeMap.values.map(_.nodeCount).sum

    assert(treeNodeCount == nodeCount)
    val interpretedPredictions: DataFrame = sf.transform(testData)
  }
}














