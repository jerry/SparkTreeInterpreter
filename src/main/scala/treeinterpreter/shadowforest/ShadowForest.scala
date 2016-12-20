package treeinterpreter.shadowforest

import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.{DataFrame, Dataset}

case class ShadowForest(treeCount: Int, treeData: Dataset[ShadowTree], nodeData: Dataset[ShadowTreeNode], numClasses: Int) {
  override def toString: String = {
    s"ShadowForest: $treeCount"
  }

  val trees: Array[ShadowTree] = treeData.collect()
  val treeRange: Range = 0 until treeCount

  val treeMap: Map[Int, ShadowTree] = trees.map(e => (e.treeID, e)).toMap

  nodeData.collect().foreach(n => {
    val targetTree = treeMap.get(n.treeId)
    if (targetTree.isDefined) {
      targetTree.get.addNode(n.nodeData)
      val v = targetTree.get.nodeCount()
      v
    } else {
      // throw an exception
    }
  })

  // TODO: This should return the observations with the predicted label, plus an array of feature names and their impact
  def transform(dataFrame: DataFrame): DataFrame = {
    dataFrame.sqlContext.emptyDataFrame
  }

  def predict(features: Vector): Double = {
    raw2prediction(predictRaw(features))
  }

  def raw2prediction(rawPrediction: Vector): Double = rawPrediction.argmax

  def transformImpl(dataset: Dataset[_]): DataFrame = {
    val bcastModel = dataset.sparkSession.sparkContext.broadcast(this)
    val predictUDF = udf { (features: Any) =>
      bcastModel.value.predict(features.asInstanceOf[Vector])
    }
    dataset.toDF() // TODO: This needs to be dataset.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
  }

  // This should probably be tuned to handle the collection and averaging of feature impact data.
  // Will return a different thing?
  def predictRaw(features: Vector): Vector = {
    // TODO: When we add a generic Bagging class, handle transform there: SPARK-7128
    // Classifies using majority votes.
    // Ignore the tree weights since all are 1.0 for now.
    val votes = Array.fill[Double](numClasses)(0.0)
    treeMap.view.foreach { treeMapping =>
      val tree = treeMapping._2

      // This part gets the votes.
      val classCounts: Array[Double] = tree.rootNode.predictImpl(features).impurityStats
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

  def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction match {
      case dv: DenseVector =>
        normalizeToProbabilitiesInPlace(dv)
        dv
      case sv: SparseVector =>
        throw new RuntimeException("Unexpected error in RandomForestClassificationModel:" +
          " raw2probabilityInPlace encountered SparseVector")
    }
  }

  def normalizeToProbabilitiesInPlace(v: DenseVector): Unit = {
    val sum = v.values.sum
    if (sum != 0) {
      var i = 0
      val size = v.size
      while (i < size) {
        v.values(i) /= sum
        i += 1
      }
    }
  }
}
