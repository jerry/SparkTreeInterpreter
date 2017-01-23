import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.rdd.RDD
import org.scalatest.FunSuite
import treeinterpreter.{DressedForest, Interp}

class TestTreeInterpreter extends FunSuite with TestSparkContext {
  test("Random Forest Regression Test") {
    val builtModel = buildModel()
    val interpRDD = Interp.interpretModel(builtModel._1, builtModel._2)
    interpRDD.collect().foreach(println)
    interpRDD.collect().foreach(item=> assert(scala.math.abs(item.checksum/item.prediction-1)<.2))
  }

  test("Dressed Forest Regression Test") {
    val builtModel = buildModel()
    val dressedForest = new DressedForest(builtModel._1)
    val item = dressedForest.interpret(builtModel._2.collect()(0))
    println(item)
    assert(scala.math.abs(item.checksum/item.prediction-1)<.2)
  }

  def buildModel(): (RandomForestModel, RDD[LabeledPoint])  = {
    val splits: Array[RDD[LabeledPoint]] = bostonData()
    val (trainingData, testData) = (splits(0), splits(1))

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

    (rf, trainingData)
  }

  def bostonData(): Array[RDD[LabeledPoint]] = {
    implicit val _sc = sc

    val currentDir = System.getProperty("user.dir")
    val replacementPath = s"$currentDir/src/test/resources/bostonData.data"
    val data: RDD[String] = _sc.textFile(replacementPath)

    val parsedData = data.filter(f => !f.equals("crim,zn,indus,chas,nox,rm,age,dis,rad,tax,ptratio,b,lstat,medv")).map(line => {
      val parsedLine = line.split(',')
      LabeledPoint(parsedLine.last.toDouble, Vectors.dense(parsedLine.dropRight(1).map(_.toDouble)))
    })

    parsedData.randomSplit(Array(0.2, 0.8), seed = 5)
  }
}


