name := "SparkTreeInterpreter"

version := "1.0.1"

//sparkVersion := "1.4.0"
//sparkVersion := "1.6.0"
sparkVersion := "2.0.0"

scalaVersion := {
  if (sparkVersion.value >= "2.0.0") {
    "2.11.8"
  } else {
    "2.10.6"
  }
}

resolvers += Resolver.mavenLocal

val sparkTestingBaseVersion = "0.4.7"

sparkComponents := Seq("core", "sql", "mllib")

libraryDependencies ++= Seq(
  "com.twitter" %% "algebird-core" % "0.9.0",
  "org.scalaz" %% "scalaz-core" % "7.1.0",
  "com.holdenkarau" %% "spark-testing-base" % s"${sparkVersion.value}_$sparkTestingBaseVersion",
  "org.json4s" %% "json4s-native" % "3.2.10",
  "org.json4s" %% "json4s-jackson" % "3.2.10",
  "org.json4s" %% "json4s-ext" % "3.2.10"
)

// TODO: Hopefully we won't need to maintain a 2.10.x of this code..
unmanagedSourceDirectories in Compile := {
  if (sparkVersion.value >= "2.0.0") {
    Seq((sourceDirectory in Compile) (_ / ("scala"))).join.value
  } else {
    Seq((sourceDirectory in Test) (_ / ("scala"))).join.value
    //Seq((sourceDirectory in Test) (_ / ("scala-2.10"))).join.value
  }
}

unmanagedSourceDirectories in Test := {
  if (sparkVersion.value >= "2.0.0") {
    Seq((sourceDirectory in Test) (_ / ("scala"))).join.value
  } else {
    Seq((sourceDirectory in Test) (_ / ("scala"))).join.value
    //Seq((sourceDirectory in Test) (_ / ("scala-2.10"))).join.value
  }
}
