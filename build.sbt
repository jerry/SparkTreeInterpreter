name := "SparkTreeInterpreter"

version := "1.0.1"

scalaVersion := "2.10.5"

resolvers += Resolver.mavenLocal

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.0.0",
  "org.apache.spark" %% "spark-mllib" % "2.0.0",
  "org.scalaz" %% "scalaz-core" % "7.1.0",
  "org.scalatest" %% "scalatest"  % "2.2.4" % "test",
  "com.holdenkarau" %% "spark-testing-base" % "2.0.0_0.4.7",
  "org.json4s" %% "json4s-native" % "3.2.10",
  "org.json4s" %% "json4s-jackson" % "3.2.10",
  "org.json4s" % "json4s-ext_2.10" % "3.2.10",
  "com.databricks" % "spark-csv_2.10" % "1.4.0"
)
