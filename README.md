=======
## Credit Card Fraud Detection
### Overview
The Annual Data Book compiled by the Federal Trade Commission reports that Credit card fraud accounted for 393,207 of the nearly 1.4 million reports of identity theft in 2020. This makes credit card fraud the second most common type of identity theft reported, behind only government documents and benefits fraud for that year. Some surveys suggest that a typical organization loses 5% of their yearly revenues to fraud. These numbers can only increase since the number of non-cash transactions increases provides more opportunities for credit card fraud.

For retailers and banks to not lose money, procedures must be put in place to detect fraud prior to it occurring.
Credit card companies must identify fraudulent credit card transactions so that customers are not charged for items that they did not purchase. To combat this problem, financial institution traditionally uses rule-based approaches to identify fraudulent transactions. These algorithms use strict rules to determine when a transaction is fraudulent.

Some challenges of a strict rule-based algorithm include:

=======
* Any new scenario that could lead to fraud needs to be manually coded into the algorithm
* Increases in customers and size of data leads to a corresponding increase in the human effort, time and cost required to track new scenarios and update the algorithm
* Since the algorithm cannot go beyond defined rules, it cannot dynamically recognize new scenarios that could result in fraudulent transaction.

To overcome these limitations, organizations are beginning to utilize machine learning and data science to build fraud detection systems. Given the size of available data, computational resources, and powerful machine learning algorithm available today, data science and machine learning processes will be able to find patterns in data and detect frauds easily.

The goal of this Credit Card Fraud Detection project is to classify a transaction as valid or fraudulent in a large dataset. Since we are dealing with discrete values, this is a binary classification problem, and we would employ the use of a supervised machine learning algorithm.
The dataset used contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced; the positive class (frauds) account for 0.172% of all transactions. 
The dataset contains only numerical input variables which are the result of a PCA transformation which was done to deidentify and anonymize the dataset for confidentiality issues. Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are ‘Time’ and ‘Amount’. 
Feature ‘Time’ contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature ‘Amount' is the transaction Amount. Feature 'Class' is the response variable, and it takes value 1 in case of fraud and 0 otherwise.

The is an imbalanced dataset. The imbalance between classes is compensated using oversampling and under sampling. The logistic regression, random forest, support vector machine, k-means are used within a cross-validation framework. Lastly, recall and accuracy are considered as metrics while choosing the best classifier.

### Control Flow
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

#### Solution Workflow
![Solution Workflow](https://user-images.githubusercontent.com/67847583/132079416-dbd29ad6-69fd-476b-9138-61d178773ba8.jpg)

#### Technology
##### Logistic Regression
Logistic regression is a classification algorithm used to find the probability of event success and event failure. It is used when the dependent variable is binary (0/1, True/False, Yes/No) in nature. It supports categorizing data into discrete classes by studying the relationship from a given set of labelled data. It learns a linear relationship from the given dataset and then introduces a non-linearity in the form of the Sigmoid function.
##### Why Logistic Regression
      1.	Logistic regression is easier to implement, interpret, and very efficient to train.
      2.	It makes no assumptions about distributions of classes in feature space.
      3.	It not only provides a measure of how appropriate a predictor (coefficient size) is, but also its direction of association (positive or negative).
      4.	Good accuracy for many simple data sets and it performs well when the dataset is linearly separable.

=======
##### Random Forest
Random forest is a technique used in modeling predictions and behavior analysis and is built on decision trees. It contains many decision trees representing a distinct instance of the classification of data input into the random forest. The random forest technique considers the instances individually, taking the one with most votes as the selected prediction.
##### Why Random Forest
      1.	It reduces overfitting in decision trees and helps to improve the accuracy
      2.	It is flexible to both classification and regression problems
      3.	It works well with both categorical and continuous values
      4.	It automates missing values present in the data
      5.	Normalizing of data is not required as it uses a rule-based approach.

=======
##### SVM
While SVMs do a good job recognizing speech, face, and images, they also do a good job at pattern recognition. Pattern recognition aims to classify data based on either a priori knowledge or statistical information extracted from raw data, which is a powerful tool in data separation in many disciplines.
##### Why SVM

      1.	SVM works relatively well when there is a clear margin of separation between classes.
      2.	SVM is effective in high dimensional spaces.
      3.	SVM can be used for other types of machine learning problems, such as regression, outlier detection, and clustering.
      4.	SVM is relatively memory efficient

=======
K-Means Clustering
K-means is a centroid-based algorithm, or a distance-based algorithm, where we calculate the distances to assign a point to a cluster. In K-Means, each cluster is associated with a centroid.
##### Why K-Means Clustering

      1.	It is easy to implement k-means and identify unknown groups of data from complex data sets. The results are presented in an easy and simple manner.
      2.	K-means algorithm can easily adjust to the changes. If there are any problems, adjusting the cluster segment will allow changes to easily occur on the algorithm.
      3.	K-means is suitable for many datasets, and it’s computed much faster than the smaller dataset. It can also produce higher clusters.
      4.	The results are easy to interpret. It generates cluster descriptions in a form minimized to ease understanding of the data.
      5.	Compared to using other clustering methods, a k-means clustering technique is fast and efficient in terms of its computational cost

=======
PgAdmin Database to store our dataset and some intermediate results
##### Database Approach
      1.	Load raw dataset into AWS S3 bucket/PgAdmin
      2.	Connect to AWS S3 bucket/PgAdmin and read data into Pandas
      3.	Load the raw data into a PgAdmin Database Instance located in AWS
      4.	Perform preprocessing steps and store cleaned data in a new table in AWS S3 bucket/PgAdmin
      5.	Store some intermediate results (which can be used later for visualization) in AWS S3 bucket/PgAdmin
      6.	The connection and S3 bucket details are in the Segment_One Jupyter Notebook.
      7.	A notebook that contains the code of the above steps is part of this repository.

##### Data Cleaning and Analysis
This project will utilize Jupyter notebook and the pandas library to perform data cleaning and analysis

### Description of communication protocols
    1. Comminucation for this project will be via a Slack Group Chat.
    2. Every team member will work in their individual branches and create a pull request which will collectively approved in the slack group chat before a designated member approves the pull request in GitHub
    3. The designated team member will then create a request to push changes to the main branch
