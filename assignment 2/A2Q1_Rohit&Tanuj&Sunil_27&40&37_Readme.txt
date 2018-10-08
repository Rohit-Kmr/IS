**KMEANS IDS**

** REQUIREMENT **

TO RUN
  1. install ancaconda navigator
  2. run the file in jupyter notebook

NOTE: please change the file name When using Currently set to "2015/01/20150101.txt"

**METHODOLOGY**

A. IN this we first do some preprocessing steps
	1. convert catagorical values to metric
	2. convert IP address to decimal
	3. convert time to metric
	4. convert malware and other detection to 0 and 1(detected)
	5. convert label(i.e our target) to 0( no attack) and 1(any attack)

B. Due to huge difference in range we then normalize the data

C. Now we select best 3 metric from the data using chi square method( which is one of the best for feature selection in sparse data set)
	and plot it using scatter3d()

D. Divide the data into training set( 80% ) and test set ( 20% )

E. Now we apply K-Means on training set

F. now we find the label of each cluster

G. now we predict the labels of test set

H. Now we create a confusion matrix and use it to find Accuracy, Precision, Recall, False Positive Rate, False Negative rate