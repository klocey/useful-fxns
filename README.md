# useful-fxns
python code for functions with general application

**Author:**   
Kenneth J. Locey, Ph.D.  
Senior Clinical Data Scientist  
Center for Quality, Safety and Value Analytics  
Rush University Medical Center  
Chicago, IL  

## Purpose
To reduce development time and redundancy by providing a validated and instructional base of source code for accomplishing a diverse set of common analytical tasks.

## Contents

Directories and files

* **101\_descriptive_stats**
	* Mean, median, mode, variance and standard error and deviation are just the tip of the iceberg when it comes to characterizing data with basic statistics. The code in this folder also calculates the weighted mean, geometric mean, harmonic mean, measures evenness/inequality, diversity, dominance, skewness, and rarity, and many more.


* **curve_fitting**
	* The most basic form of testing models and making predictions. The code in this folder allows the user to fit and test a wide variety of models to cumulative and non-cumulative data, time-series data, histograms and frequency distributions, and rank-value curves. The source code also includes metrics to quantify and compare the performance of curve fitting models.


* **diagnostic_metrics**
	* Binary classification, predicting yes/no outcomes, is a popular form of predictive modeling. The source code in the folder contains functions for generating and analyzing receiver operating characteristics, precision-recall curves, and their variants, while providing greater insight, control, and customizability than functions included in canned packages. 

* **GIS**
	* The use of geograhical information systems can be as simple as calculating the distance between two points on Earth and making a map, or as complicated as analyzing the spatial autocorrelation among geographical features. The code in this folder provides functions for accomplishing these tasks and many more.
	
* **hypothesis_testing**
	* Asking whether two or more samples are significantly different is a basic statistical task with many options available and many assumptions to be met. The source code in this folder provides functions for testing normality, linearity, etc., and for testing whether two or more samples are significantly different.  

* **machine_learning**
	* Machine learning predates modern computers but has only recently seen a surge in popularity in healthcare. It is now a go-to for predictive analytics, for formal predictive frameworks, for clustering and for reducing the complexity of modern data sets. The code in this repository provides functions for customizing and running clustering algorithms, dimensionality reduction, regression modeling, data scaling, feature elimination, cross-validation, anomaly detection and for using artificial neural networks, decision trees, and random forests. 

* **pandas**
	* Pandas is python's core library for handling data structures (dataframes). It's also super handy for feature engineering for reading/writing files, for data serialization (e.g., to and from json), for conducting GIS, for dealing with times and dates, and many other tasks beyond simply manipulating tables, rows, and columns. The code in this folder will assist the user with basic pandas tasks and with its more advanced features.
	 
* **plotting_figures**
	* Users of plotting libraries often struggle to go beyond the defaults of basic plotting functions. The code in this folder will assist the user in producing novel and customized figures in both matplotlib and plotly. These two libraries are included to the exclusion of others (Seaborn) for two reasons. Matplotlib is huge and offers incredible control over custom plotting. Plotly offers less control but easily accomplished what matplotlib is not intended for, interactive charts and tables.

* **resampling**
	* Random resampling of data is a common analytical task for generating confidence limits, testing the dependence of outcomes on sample size and sampling error, and for generating predictions of greater reliability. The code in this folder will assist the user with the task of uniform and stratified random sampling and repeated random sampling (bootstrapping).