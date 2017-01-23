package treeinterpreter

import treeinterpreter.TreeInterpretationTools.{Feature, NodeContribution}

case class TreeInterpretation(bias: Double, prediction: Double, contributions: NodeContribution, treeCount: Double = 1.0, checksum: Double = 0.0) {
  override def toString: String = {
    s"""
       | bias: $bias
       | prediction: $prediction
       | contributionMap $contributions
       | sumOfTerms: $checksum""".stripMargin
  }

  def plus(r: TreeInterpretation): TreeInterpretation = {
    val leftContributions: Set[Feature] = contributions.keySet
    val rightContributions: Set[Feature] = r.contributions.keySet

    val commonValues: Array[(Feature, Double)] = leftContributions.intersect(rightContributions).map(i => {
      val rightValue = r.contributions.get(i)
      val leftValue = contributions.get(i)
      val amount = leftValue.get + rightValue.get
      (i, amount)
    }).toArray

    val leftValues: Array[(Feature, Double)] = leftContributions.diff(rightContributions).map(i => {
      val leftValue = contributions.get(i)
      val amount = leftValue.get
      (i, amount)
    }).toArray

    val rightValues: Array[(Feature, Double)] = rightContributions.diff(leftContributions).map(i => {
      val rightValue = r.contributions.get(i)
      val amount = rightValue.get
      (i, amount)
    }).toArray

    val allValues: NodeContribution = (commonValues ++ leftValues ++ rightValues).toMap

    TreeInterpretation(
      bias + r.bias,
      prediction + r.prediction,
      allValues,
      treeCount + r.treeCount)
  }
}
