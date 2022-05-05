name := "spark-infotheoretic-feature-selection"

version := "0.1-spark-3.0.1"

organization := "com.github.sramirez"

scalaVersion := "2.12.2"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-mllib" % "3.0.1" % Provided,
  "joda-time" % "joda-time" % "2.10.5" % Provided
)

libraryDependencies ++= Seq(
  "org.scalatest" %% "scalatest" % "3.0.8",
  "junit" % "junit" % "4.12"
).map(_ % Test)

resolvers += "Maven Repo" at "https://repo1.maven.org/maven2/"

publishMavenStyle := true


