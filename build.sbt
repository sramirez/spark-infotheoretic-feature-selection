sbt.version := "0.13.8"

name := "spark-infotheoretic-feature-selection"

version := "0.1"

organization := "com.github.sramirez"

scalaVersion := "2.10.4"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.3.0"

resolvers ++= Seq(
  "Apache Staging" at "https://repository.apache.org/content/repositories/staging/",
  "Typesafe" at "http://repo.typesafe.com/typesafe/releases",
  "Local Maven Repository" at "file://"+Path.userHome.absolutePath+"/.m2/repository"
)

publishMavenStyle := true

sparkPackageName := "sramirez/infotheoretic-feature-selection"

sparkVersion := "1.3.0"

sparkComponents += "mllib"

