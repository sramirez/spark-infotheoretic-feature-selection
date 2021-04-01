name := "spark-infotheoretic-feature-selection"

version := "0.1"

organization := "com.github.sramirez"

scalaVersion := "2.11.12"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.2.0" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.2.0" % "provided"
libraryDependencies += "org.scalatest" %% "scalatest" % "2.2.6"  % "test"
libraryDependencies += "joda-time" % "joda-time" % "2.9.4" % "test"
libraryDependencies += "junit" % "junit" % "4.12" % "test"
