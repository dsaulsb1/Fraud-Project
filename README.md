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
      3.	It not only provides a measure of how appropriate a predictor (coefficient size) is, but also its direction of association 
      (positive or negative).
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
      4.	SVM is relatively memory efficient.

=======

K-Means Clustering
K-means is a centroid-based algorithm, or a distance-based algorithm, where we calculate the distances to assign a point to a cluster. In K-Means, each cluster is associated with a centroid.
##### Why K-Means Clustering
      1.	It is easy to implement k-means and identify unknown groups of data from complex data sets. 
            - The results are presented easily and straightforwardly.
      2.	K-means algorithm can easily adjust to the changes. 
            - If there are any problems, adjusting the cluster segment will allow quick changes to the algorithm.
      3.	K-means easily scales to large datasets.
      4.	K-Means returns clusters that can be easily interpreted and even visualized. 
            - This simplicity makes it highly useful in some cases when you need a quick overview of the data segments.
      5.	Compared to using other clustering methods, a k-means clustering is fast and efficient in terms of its computational cost

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
    2. Every team member will work in their individual branches.
    3. Team members will create pull requests which will be collectively approved in the slack channel.
    4. A designated team member will validate the pull request, and merge the request to the main branch.
    
    
### Results
#### Exploratory Data Analysis
##### Univariate Analysis:
      1. Univariate plots show that the dataset is highly imbalanced. 
      2. The pie chart shows an imbalance in the data, with only 0.17% of the total cases being fraudulent.
      3. The univariate distribution plot of the time and amount feature show we have a dataset with some large outlier values 
      for amount.
      4. The time feature is distributed across two days
      5. Bivariate plots show that the valid transaction class has a normal distribution shape across most of the features.
      6. Conversely, the fraud class show long-tailed distribution across many of the features.
 ##### Univariate Analysis Results    
![Result_Pie_Chart](https://user-images.githubusercontent.com/67847583/133525687-7eb8eac0-35ef-426a-837d-e228442f3d98.png)
![Univariate_Analysis_Time_Amount_Distribution](https://user-images.githubusercontent.com/67847583/133525463-3a09f744-49c0-4924-9726-2862e5972075.png)

##### Bivariate Analysis
![Bivariate_Analysis_Distr_Plots](https://user-images.githubusercontent.com/67847583/133525502-4c439bfe-36bc-411f-9f60-f6a026ff7d60.png)

#### Naive Model Results
      1. While the naive logistic classifier accuracy is 100%, our classifier did not do an excellent job at predicting fraudulent 
      transactions. 
      2. With precision and recall of 0.84 and 0.62, we would need a better understanding of the dataset to determine the best way to 
      improve the recall metric.
      3. While the naive random forest classifier accuracy is 100%, and precision is 95%, our random forest classifier only achieved a 
      77% recall. 
      4. We would need a better understanding of the dataset to determine the best way to improve the recall metric.
      
###### Naive Model Results
![Naive_Model_Results](https://user-images.githubusercontent.com/67847583/133527239-4550e302-88ea-4280-87b3-3199f31992f1.png)

#### The ROC-AUC Curve
      1. AUC - ROC curve is a performance measurement for a classifier at various classification thresholds.
      2. ROC is a probability curve that plots True Positives and False Positives
      3. An ROC curve plots TPR vs. FPR at different classification thresholds.
      4. Lowering the classification threshold classifies more items as positive, thus increasing both False Positives 
      and True Positives. 
      5. Increasing the classification threshold classifies more items as negative, thus increasing both False Negatives 
      and True Negatives.
      6. The AUC lets us find the optimal classification threshold that minimizes false positives and false negatives.
      7. For our credit card classification problem, we would want a classification threshold that increases true positives.
      8. AUC measures the entire two-dimensional area underneath the entire ROC curve from (0,0) to (1,1).
      9. AUC tells how much the model is capable of distinguishing between classes.
      10. Higher the AUC, the better the model is at predicting 0 classes as 0 and 1 classes as 1.
      
#### The Learning Curve
      1. The learning curve is the plot of the training/cross-validation error versus the sample size.
      2. Learning curves show the relationship between training set size and the recall metric on the training and validation sets.
      3. The learning curve detects whether the model has the high bias or high variance.
      4. If the model suffers from high bias problem, as the sample size increases, training error will increase and the 
      cross-validation error will decrease.
      5. Training error and cross-validation error will end up close to each other but still at a high error rate.
      4. If the model suffers from high variance, as the sample size increases, the training error will keep increasing and 
      cross-validation error will keep decreasing.
      5. Training error and cross-validation error will end up at a low training and cross-validation error rate.

#### Undersampling Model Results
    1. By Undersampling our the majority class in our dataset, all classifiers achieved recall scores greater than 
    85% with the exception of the Support vector classifier.
    2. The ROC Curve show that the Support Vector Classifier has the largest AUC, while the decision tree
    classifier has the smallest AUC
    3. All undersampling curves show a fairly ideal learning curve. 
    4. As the training size increases, training error and validation error generally reduces and end up at a low rate.
      
###### Undersampling Results
![Model_Performances_Undersampling](https://user-images.githubusercontent.com/67847583/133974705-93c65293-df1e-426d-9e57-2a1e69180ceb.png)
![ROC_Curve_Undersampling](https://user-images.githubusercontent.com/67847583/133974457-d0f25b8b-b0ef-412b-9173-547e3080f906.png)
![Learning_Curve_Undersampling](https://user-images.githubusercontent.com/67847583/133974490-d26feacf-3707-4d04-8e78-56eb09370ab1.png)

#### Oversampling Model Results
    1. By Oversampling the dataset, we ahieved recall scores greater than 85% for all classifiers. The Random Forest classifier had 
    the best accuracy of 99%
    2. The ROC Curve show that the random forest classifier has the largest AUC while the decision tree classifier has the smallest AUC
    3. All oversampling curves show a fairly ideal learning curve. 
    4. As the training size increases, training error and validation error generally reduces and end up at a low error rate
      
###### Comparing Model Performances
![Comparing_Model_Results](https://user-images.githubusercontent.com/67847583/133974960-19bb87bd-9c25-4141-bbcb-815b9d4efc7d.png)
![ROC_Curve_Oversampling](https://user-images.githubusercontent.com/67847583/133974982-e5f61867-5b71-4938-bf86-ba2093008436.png)
![Learning_Curve_Oversampling](https://user-images.githubusercontent.com/67847583/133975015-f1849656-26b9-41d8-84e0-94e078a67c46.png)


#### Dashboard
[Link to Dashboard](https://public.tableau.com/views/Segment_Two_Dashboard_20210913/FraudDetectionStory?:language=en-US&publish=yes&:display_count=n&:origin=viz_share_link "Link to Dashboard") 


#### Presentation
[Link to Presentation](https://docs.google.com/presentation/d/1dZFGsjNqBm8g_7548PAecGgwXyDAJgDpwp7bEyCh3jY/edit?usp=sharing "Link to Presentation")

### Summary
The Dataset

     1. The dataset used for this project has 284807 rows of credit card transactions. 
     2. Exploratory data analysis reveal as expected that we have a highly imbalanced dataset.
     3. Only 0.17% of all transaction are fraudulent.
     4. While a large portion of the features have been anonymized with PCA, univariate and bivariate 
     distribution plots show that the genuine transaction class has an approximately normal distribution 
     across all features, and the fraud class was had a left skewed distribution for many of the features.

Naive Models

    1. While naive logistic regression and random forest had an accuracy of 100% and a precisions of 
    84% and 96% respectively, both classifiers only managed recall scores of 62% and 77% respectively.
    2. This means that, the classifiers would miss fraud transaction almost 25% of the time. 
    3. This type of metric would cost an organization alot of money.

Performance Metrics

    1. Classifying transactions as fraud or genuine is an anomaly detection problem where only a small 
    fraction are the anomalies, measuring model performance with the accuracy metric will not be ideal.
    2. To capture fraud transactions we would require a classifier that has a high recall metric.
    3. Recall is the ratio of of True Positives to the total of True Positives and False Positives

Oversampling, Undersamling, ROC-AUC Curve, and the Learning Curve

    1. To improve the recall score of the naive models, we employ oversampling and underampling.
    2. With these methods, we achieved recall scores greater than 90%, and 85% for the undersampling 
    and oversampling methods respectivelly.
    3. While recall for random forest was highest at 95.9%, the classifier had a lower AUC value 
    (91.5) than the logistic regression classifier with AUC of 92.1.
    4. Analysis of the learning curve show that the logistic regression had a generally good fit.

#### Best Model
To choose the best model, we may consider the following factors:

Characteristics of the dataset

    1. Since we are dealing with an imbalanced dataset, our first intuition is apply techniques such as 
    undersampling and oversampling
    2. The Random Forest Classifier works well with resampling. 
    3. The Random Forest classifier lets us bootstrap samples, so we take a resample of our training data set. 
    4. And then we rebuild classification or regression trees on each of those bootstrap samples. 

Evaluation Metric: Recall Score

    1. Since imbalanced datasets will generally have high accuracy scores, we need a different metric 
    to evaluate model performance.
    2. The choice of model may depend on the recall metric which measures the ratio of of True Positives 
    to the total of True Positives and False Positives

AUC-ROC Curve

    1. The AUC-ROC curve helps us determine the optimal classification threshold that 
    minimizes False Positives, and False Negatives.
    2. The AUC-ROC curve can be used in addition to the recall metric to select the best model 
    for this classification problem

Learning Curve

    1. The learning curve helps us evaluate whether our model is overfitting, underfitting, or has a good fit.
    2. The learning curve can be used in addition to the recall metric, and the AUC-ROC curve to select the 
    best model for this classification problem.
      

#### Recommendation
    1. One challenge with this project was computation resources required to run the RandomizedGridSearchCV and the model Cross-Validation scores.
     - One way to to mitigate this challenge in the future may be to use the HalvingGridSearchCV which may in some cases may be 
     30% faster than the RandomizedGridSearchCV.
     - We may also explore using an online environment that has unlimited computation resources that can handle the resource 
     requirements for memory and CPU intensive models  and processes. 
    2. Since feature extraction had been done on the dataset, visualizing potentially interesting relationships was not possible with this dataset.
