# Credit Card Fraud Detection

## Communication protocols

- We created a Slack Group Chat in order to collaborate quickly and efficiently.

# Project Description:

## Problem Statement : 
Credit card companies must identify fraudulent credit card transactions so that customers are not charged for items that they did not purchase. The dataset used contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced; the positive class (frauds) account for 0.172% of all transactions. 

The imbalance between classes is compensated using oversampling and under sampling. The logistic regression, random forest, support vector machine, k-means are used within a cross-validation framework. Lastly, recall and accuracy are considered as metrics while choosing the best classifier.


## Control Flow

1.	Understanding the problem
2.	Importing required libraries and understanding their use
3.	Importing data and learning its structure
4.	Performing basic EDA
5.	Scaling different variables
6.	Outlier treatment
7.	Building basic Classification model with Random Forest
8.	Nearmiss technique for under sampling data
9.	SMOTE for oversampling data
10.	cross validation in the context of under sampling and oversampling
11.	Pipelining with sklearn/imblearn
12.	Applying Linear model: Logistic Regression
13.	Applying Ensemble technique: Random Forest
14.	Applying Non-Linear Algorithms: Support Vector Machine, Decision Tree, and k-Nearest Neighbor
15.	Making predictions on test set and computing validation metrics
16.	ROC curve and Learning curve
17.	Comparison of results and Model Selection
18.	Visualization with seaborn and matplotlib

## Technology

### Logistic Regression

Logistic regression is a classification algorithm used to find the probability of event success and event failure. It is used when the dependent variable is binary (0/1, 
True/False, Yes/No) in nature. It supports categorizing data into discrete classes by studying the relationship from a given set of labelled data. It learns a linear relationship 
from the given dataset and then introduces a non-linearity in the form of the Sigmoid function.

####  Why Logistic Regression
      1.	Logistic regression is easier to implement, interpret, and very efficient to train.
      2.	It makes no assumptions about distributions of classes in feature space.
      3.	It not only provides a measure of how appropriate a predictor (coefficient size) is, but also its direction of association (positive or negative).
      4.	Good accuracy for many simple data sets and it performs well when the dataset is linearly separable.

### Random Forest

Random forest is a technique used in modeling predictions and behavior analysis and is built on decision trees. It contains many decision trees representing a distinct instance of 
the classification of data input into the random forest. The random forest technique considers the instances individually, taking the one with most votes as the selected 
prediction.

#### Why Random Forest
      1.    It reduces overfitting in decision trees and helps to improve the accuracy
      2.	It is flexible to both classification and regression problems
      3.	It works well with both categorical and continuous values
      4.	It automates missing values present in the data
      5.	Normalizing of data is not required as it uses a rule-based approach.


### SVM

While SVMs do a good job recognizing speech, face, and images, they also do a good job at pattern recognition. Pattern recognition aims to classify data based on either a priori 
knowledge or statistical information extracted from raw data, which is a powerful tool in data separation in many disciplines.

#### Why SVM

      1.	SVM works relatively well when there is a clear margin of separation between classes.
      2.	SVM is effective in high dimensional spaces.
      3.	SVM can be used for other types of machine learning problems, such as regression, outlier detection, and clustering.
      4.	SVM is relatively memory efficient

### K-Means Clustering

K-means is a centroid-based algorithm, or a distance-based algorithm, where we calculate the distances to assign a point to a cluster. In K-Means, each cluster is associated with a centroid.

#### Why K-Means Clustering
      1.	It is easy to implement k-means and identify unknown groups of data from complex data sets. The results are presented in an easy and simple manner.
      2.	K-means algorithm can easily adjust to the changes. If there are any problems, adjusting the cluster segment will allow changes to easily occur on the algorithm.
      3.	K-means is suitable for many datasets, and it’s computed much faster than the smaller dataset. It can also produce higher clusters.
      4.	The results are easy to interpret. It generates cluster descriptions in a form minimized to ease understanding of the data.
      5.	Compared to using other clustering methods, a k-means clustering technique is fast and efficient in terms of its computational cost

### PgAdmin Database to store our dataset and some intermediate results

### Database Approach
      1.	Load raw dataset into PgAdmin
      2.	Connect to PgAdmin and read data into Pandas
      3.	Perform preprocessing steps and store cleaned data in a new table in PgAdmin
      4.	Store some intermediate results (which can be used later for visualization) in PgAdmin
