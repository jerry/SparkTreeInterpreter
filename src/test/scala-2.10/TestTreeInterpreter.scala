import com.holdenkarau.spark.testing.SharedSparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.scalatest.FunSuite
import treeinterpreter.{DressedForest, Interp}

class TestTreeInterpreter extends FunSuite with TestSparkContext {
  test("Random Forest Regression Test") {
    implicit val _sc = sc

    val currentDir = System.getProperty("user.dir")
    val replacementPath = s"$currentDir/src/test/resources/bostonData.data"
    val data = _sc.textFile(replacementPath)

    val parsedData = data.map(line => {
      val parsedLine = line.split(',')
      LabeledPoint(parsedLine.last.toDouble, Vectors.dense(parsedLine.dropRight(1).map(_.toDouble)))
    })

    val splits = parsedData.randomSplit(Array(0.2, 0.8), seed = 5)
    val (trainingData, testData) = (splits(0), splits(1))

    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 10
    val featureSubsetStrategy = "auto"
    val impurity = "variance"
    val maxDepth = 2
    val maxBins = 32

    val classimpurity = "gini"

    val rf: RandomForestModel = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, seed = 21)

    val labelsAndPredictions = trainingData.map { point =>
      val prediction = rf.predict(point.features)
      (point.label, prediction)
    }

    val testMSE = math.sqrt(labelsAndPredictions.map { case (v, p) => math.pow(v - p, 2) }.mean())

    println("Test Mean Squared Error = " + testMSE)

    val interpRDD = Interp.interpretModel(rf, trainingData)

    interpRDD.collect().foreach(println)

    interpRDD.collect().foreach(item=> assert(scala.math.abs(item.checksum/item.prediction-1)<.2))
  }

  test("Dressed Forest Regression Test") {
    implicit val _sc = sc


    val currentDir = System.getProperty("user.dir")
    val replacementPath = s"$currentDir/src/test/resources/bostonData.data"
    val data = _sc.textFile(replacementPath)

    val parsedData = data.map(line => {
      val parsedLine = line.split(',')
      LabeledPoint(parsedLine.last.toDouble, Vectors.dense(parsedLine.dropRight(1).map(_.toDouble)))
    })

    val splits = parsedData.randomSplit(Array(0.2, 0.8), seed = 5)
    val (trainingData, testData) = (splits(0), splits(1))

    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 10
    val featureSubsetStrategy = "auto"
    val impurity = "variance"
    val maxDepth = 2
    val maxBins = 32

    val classimpurity = "gini"

    val rf: RandomForestModel = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, seed = 21)

    val labelsAndPredictions = trainingData.map { point =>
      val prediction = rf.predict(point.features)
      (point.label, prediction)
    }

    val testMSE = math.sqrt(labelsAndPredictions.map { case (v, p) => math.pow(v - p, 2) }.mean())

    println("Test Mean Squared Error = " + testMSE)

    val dressedForest = new DressedForest(rf)
    val item = dressedForest.interpret(trainingData.collect()(0))
    println(item)
    assert(scala.math.abs(item.checksum/item.prediction-1)<.2)
  }
}


