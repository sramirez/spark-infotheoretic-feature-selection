An Information Theoretic Feature Selection Framework
=====================================================

The present framework implements Feature Selection (FS) on Spark for its application on Big Data problems. This package contains a generic implementation of greedy Information Theoretic Feature Selection methods. The implementation is based on the common theoretic framework presented in [1]. Implementations of mRMR, InfoGain, JMI and other commonly used FS filters are provided. In addition, the framework can be extended with other criteria provided by the user as long as the process complies with the framework proposed in [1].

Spark package: http://spark-packages.org/package/sramirez/spark-infotheoretic-feature-selection

Please cite as: S. Ramírez-Gallego; H. Mouriño-Talín; D. Martínez-Rego; V. Bolón-Canedo; J. M. Benítez; A. Alonso-Betanzos; F. Herrera, "An Information Theory-Based Feature Selection Framework for Big Data Under Apache Spark," in IEEE Transactions on Systems, Man, and Cybernetics: Systems, in press, pp.1-13, doi: 10.1109/TSMC.2017.2670926
URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7970198&isnumber=6376248


## Main features:

* Version for new ml library.
* Support for sparse data and high-dimensional datasets (millions of features).
* Improved performance (less than 1 minute per iteration for datasets like ECBDL14 and kddb with 400 cores).

This work has associated two submitted contributions to international journals which will be attached to this request as soon as they are accepted. This software has been proved with two large real-world datasets such as:

- A dataset selected for the GECCO-2014 in Vancouver, July 13th, 2014 competition, which comes from the Protein Structure Prediction field (http://cruncher.ncl.ac.uk/bdcomp/). We have created a oversampling version of this dataset with 64 million instances, 631 attributes, 2 classes.
- kddb dataset: http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#kdd2010%20%28bridge%20to%20algebra%29. 20M instances and almost 30M of attributes.

## Example (ml): 
	import org.apache.spark.ml.feature._
	val selector = new InfoThSelector()
		.setSelectCriterion("mrmr")
	      	.setNPartitions(100)
	      	.setNumTopFeatures(10)
	      	.setFeaturesCol("features")
	      	.setLabelCol("class")
	      	.setOutputCol("selectedFeatures")
   
	val result = selector.fit(df).transform(df)

## Example (MLLIB): 
	import org.apache.spark.mllib.feature._
	val criterion = new InfoThCriterionFactory("mrmr")
	val nToSelect = 100
	val nPartitions = 100
	
	println("*** FS criterion: " + criterion.getCriterion.toString)
	println("*** Number of features to select: " + nToSelect)
	println("*** Number of partitions: " + nPartitions)
	
	val featureSelector = new InfoThSelector(criterion, nToSelect, nPartitions).fit(data)
	
	val reduced = data.map(i => LabeledPoint(i.label, featureSelector.transform(i.features)))
	reduced.first()
        

Design doc: https://docs.google.com/document/d/1HOaPL_HJzTbL2tVdzbTjhr5wxVvPe9e-23S7rc2VcsY/edit?usp=sharing

## Prerequisites:

LabeledPoint data must be discretized as integer values in double representation, ranging from 0 to 255. 
By doing so, double values can be transformed to byte directly thus making the overall selection process much more efficient (communication overhead is deeply reduced).

Please refer to the MDLP package if you need to discretize your dataset: 

https://spark-packages.org/package/sramirez/spark-MDLP-discretization

## Contributors

- Sergio Ramírez-Gallego (sramirez@decsai.ugr.es) (main contributor and maintainer).
- Héctor Mouriño-Talín (h.mtalin@udc.es)
- David Martínez-Rego (dmartinez@udc.es)

## References

[1] Brown, G., Pocock, A., Zhao, M. J., & Luján, M. (2012). "Conditional likelihood maximisation: a unifying framework for information theoretic feature selection." The Journal of Machine Learning Research, 13(1), 27-66.


