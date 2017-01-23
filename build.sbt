name := "SparkTreeInterpreter"

version := "2.0.0"

scalaVersion := "2.11.8"

resolvers += Resolver.mavenLocal

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.0.0",
  "org.apache.spark" %% "spark-mllib" % "2.0.0",
  "com.holdenkarau" %% "spark-testing-base" % "2.0.0_0.4.7"
)
