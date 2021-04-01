name := "spark-infotheoretic-feature-selection"

version := "0.1"

organization := "com.github.sramirez"

scalaVersion := "2.12.12"

libraryDependencies += "org.apache.spark" %% "spark-core" % "3.0.1" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.0.1" % "provided"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.1"  % "test"
libraryDependencies += "joda-time" % "joda-time" % "2.9.4" % "test"
libraryDependencies += "junit" % "junit" % "4.12" % "test"
