package org.apache.spark.ml.tree

import org.apache.spark.ml.linalg.{Vector, DenseVector}

object ImpactExtractor {
  case class LabeledFeatureImpact(featureIndex: Int, fieldName: String, featureImpact: Double, featureValue: Double) {
    override def toString: String = s"""{
                                       |      "feature_index": $featureIndex,
                                       |      "field_name": "$fieldName",
                                       |      "feature_impact": $featureImpact,
                                       |      "feature_value": $featureValue
                                       |    }""".stripMargin
  }

  def json(impacts: Array[LabeledFeatureImpact]): String = {
    val impactJsonStringArray: Array[String] = impacts.map(f => f.toString)
    println(impactJsonStringArray)
    s"""{
       |  "contributions": [
       |    ${impactJsonStringArray.mkString(",\n    ")}
       |  ]
       |}""".stripMargin
  }

  def labeledContributionsAsJson(interpretedPrediction: DenseVector, featureList: String, features: Vector): String = {
    val jsonOutput = json(labeledContributions(interpretedPrediction, featureList, features))

    jsonOutput.toString
  }

  def labeledContributions(interpretedPrediction: DenseVector, featureList: String, features: Vector): Array[LabeledFeatureImpact] = {
    var nextIndex = 0
    featureList.split(",").map(f => {
      val featureValues = features.toArray
      val lfi = LabeledFeatureImpact(nextIndex, f, interpretedPrediction.values(nextIndex), featureValues(nextIndex))
      nextIndex += 1
      lfi
    })
  }

  /*
    sparkSession.udf.register("labeledContributionsAsJson", (contributions: DenseVector, featureList: String, features: DenseVector) =>
      ImpactExtractor.labeledContributions(contributions, featureList, features))
   */
}
