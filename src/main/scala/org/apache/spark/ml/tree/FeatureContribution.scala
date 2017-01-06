package org.apache.spark.ml.tree

case class FeatureContribution(featureIndex: Int, contribution: Array[Double]) {
  override def toString: String = s"FeatureContribution: featureIndex: $featureIndex, contribution: [${contribution.mkString(", ")}]"
}