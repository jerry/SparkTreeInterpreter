package org.apache.spark.ml.tree

/** Container for feature contribution information
  *
  * @param featureIndex integer indicator of the feature's index in the array of all feature names and the vector of feature values
  * @param contribution amount of change caused by this feature for each possible class in the model
  */
case class FeatureContribution(featureIndex: Int, contribution: Array[Double]) {
  override def toString: String = s"FeatureContribution: featureIndex: $featureIndex, contribution: [${contribution.mkString(", ")}]"
}