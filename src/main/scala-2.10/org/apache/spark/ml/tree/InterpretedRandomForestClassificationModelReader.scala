package org.apache.spark.ml.tree

import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, RandomForestClassificationModel}
import org.apache.spark.ml.util.DefaultParamsReader.Metadata
import org.apache.spark.ml.util.{DefaultParamsReader, MLReader}
import org.json4s.DefaultFormats

class InterpretedRandomForestClassificationModelReader extends MLReader[InterpretedRandomForestClassificationModel] {

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
    require(numTrees == trees.length, s"InterpretedRandomForestClassificationModel.load expected $numTrees" +
      s" trees based on metadata but found ${trees.length} trees.")

    val model = new InterpretedRandomForestClassificationModel(metadata.uid, trees, numFeatures, numClasses)
    DefaultParamsReader.getAndSetParams(model, metadata)
    model
  }
}