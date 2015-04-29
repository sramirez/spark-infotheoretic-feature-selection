An Information Theoretic Feature Selection Framework
=====================================================

The present framework implements Feature Selection (FS) on Spark for its application on Big Data problems. This package contains a generic implementation of greedy Information Theoretic Feature Selection methods. The implementation is based on the common theoretic framework presented in [1]. Implementations of mRMR, InfoGain, JMI and other commonly used FS filters are provided. In addition, the framework can be extended with other criteria provided by the user as long as the process complies with the framework proposed in [1].

## Main features:
* Support for sparse data (release 0.1).
* Pool optimization.
* Improved performance from previous version.

This work has associated two submitted contributions to international journals which will be attached to this request as soon as they are accepted. This software has been proved with two large real-world datasets such as:

- A dataset selected for the GECCO-2014 in Vancouver, July 13th, 2014 competition, which comes from the Protein Structure Prediction field (http://cruncher.ncl.ac.uk/bdcomp/). The dataset has 32 million instances, 631 attributes, 2 classes, 98% of negative examples and occupies, when uncompressed, about 56GB of disk space.
- Epsilon dataset: http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#epsilon. 400K instances and 2K attributes.

## Example: 

	val criterion = new InfoThCriterionFactory("mrmr")
	val nToSelect = 100
	val nPool = 100 // 0 -> w/o pool
	
	println("*** FS criterion: " + criterion.getCriterion.toString)
	println("*** Number of features to select: " + nToSelect)
	println("*** Pool size: " + nPool)
	
	val featureSelector = InfoThSelector.train(criterion, 
	      	data, // RDD[LabeledPoint]
      		nToSelect, // number of features to select
	      nPool) // number of features in pool
    	featureSelector
	
	val reduced = data.map(i => LabeledPoint(i.label, featureSelector.transform(i.features)))
	reduced.first()
        

Design doc: https://docs.google.com/document/d/1HOaPL_HJzTbL2tVdzbTjhr5wxVvPe9e-23S7rc2VcsY/edit?usp=sharing

## Prerequisites:

Data must be discretized. Each LabeledPoint has to be a Integer in range [0, 255], despite of using the Double format.
It can be done using MDLP discretizer for Spark, which can be found in: http://spark-packages.org/package/sramirez/spark-MDLP-discretization.
This reason behind this decision is to achive a better performance in the whole process of selection by simplifying the storing and the transferring of complex data.

## Contributors

- Sergio Ramírez-Gallego (sramirez@decsai.ugr.es) (main contributor and maintainer).
- Héctor Mouriño-Talín (h.mtalin@udc.es)
- David Martínez-Rego (dmartinez@udc.es)

##References

[1] Brown, G., Pocock, A., Zhao, M. J., & Luján, M. (2012). 
"Conditional likelihood maximisation: a unifying framework for information theoretic feature selection." 
The Journal of Machine Learning Research, 13(1), 27-66.
