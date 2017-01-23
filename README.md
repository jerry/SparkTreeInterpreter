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

To run tests, just run `sbt test`.


ML Usage and Tests
-----
InterpretedRandomForestClassifier extends RandomForestClassifier and can be used as a drop-in replacement.

InterpretedRandomForestRegressor extends RandomForestRegressor and can be used as a drop-in replacement.
