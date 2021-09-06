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



