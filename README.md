===============================
Spark Tree Interpreter (Scala & Spark ML/MLlib port of https://github.com/andosa/treeinterpreter)
===============================

This is a Scala & Spark ML/MLlib port of https://github.com/andosa/treeinterpreter.

Free software: BSD license

Dependencies
------------
Spark 2.0.0+


MLlib Usage and Tests
-----
Given a trained RandomForestModel/DecisionTreeModel and an RDD[LabeledPoint], we have

```
Interp.interpretModel(model, trainingData)
```
yields
``` 
 prediction:20.897107560022445
 bias:22.05948146469634
 contributionMap Map(Some(4) -> 0.38498427854727063, Some(6) -> 0.6947299575790253, Some(12) -> -0.7793909217869917, Some(5) -> -0.9248027735005142, Some(2) -> -1.01193915560854)
 sumOfTerms: 20.42306284992659
```

The sum of bias and feature contributions should equal the prediction, but due to floating point arithmetic they will be slightly off.


ML Usage and Tests
-----
Both InterpretedRandomForestClassifier (which extends RandomForestClassifier) and InterpretedRandomForestRegressor (which extends RandomForestRegressor) can be used as drop-in replacements in your SparkML pipeline.

The dataset returned by `transform` will include 3 additional fields:
 
* bias: double (nullable = true) - representing the average bias across the root nodes in the trees
* checksum: double (nullable = true) - the sum of contributions and the bias, for use in validation
* contributions: vector (nullable = true) - the individual contribution values, for each feature

InterpretedRandomForestClassifier adds a fourth field:

* predictedLabelProbability: double (nullable = true) - the probability of the predicted label

Tests
-----
To run tests, just run `sbt test`.

