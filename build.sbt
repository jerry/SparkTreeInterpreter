name := "SparkTreeInterpreter"

version := "2.0.0"

sparkVersion := "2.0.0"

// Has been tested against 2.1
//sparkVersion := "2.1.0"

scalaVersion := {
  if (sparkVersion.value >= "2.0.0") {
    "2.11.8"
  } else {
    "2.10.6"
  }
}

resolvers += Resolver.mavenLocal

val sparkTestingBaseVersion = "0.5.0"

sparkComponents := Seq("core", "sql", "mllib")

libraryDependencies ++= Seq(
  "com.holdenkarau" %% "spark-testing-base" % s"${sparkVersion.value}_$sparkTestingBaseVersion"
)

// Hopefully we won't need to maintain a pre Spark 2.0 version of this, but leaving the logic around just in case
unmanagedSourceDirectories in Compile := {
  if (sparkVersion.value >= "2.0.0") {
    Seq((sourceDirectory in Compile) (_ / ("scala"))).join.value
  } else {
    Seq((sourceDirectory in Test) (_ / ("scala-2.10"))).join.value
  }
}

unmanagedSourceDirectories in Test := {
  if (sparkVersion.value >= "2.0.0") {
    Seq((sourceDirectory in Test) (_ / ("scala"))).join.value
  } else {
    Seq((sourceDirectory in Test) (_ / ("scala-2.10"))).join.value
  }
}
