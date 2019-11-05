# Predictive Model

This is the Data Science Project in [SharpestMinds](https://www.sharpestminds.com/).

#### Project Aims:

- Predict customer's behavior to retain them
- Compare different methods and find the best one
- Work on an on-hand data science project end-to-end.

#### Data Source:

I get dataset from [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn). 

After reading [Hands-on: Predict Customer Churn](https://towardsdatascience.com/hands-on-predict-customer-churn-5c2a42806266), I create my own predictive models for customer churn.

#### Data Preprocessing:

1. Replace binary variables with [0,1]
2. Change categorical data into numerical data
3. Drop missing data
4. Split data into training set and testing set (80% / 20%)
5. Store the above two datasets. (For small sets and **consistent experimentation**)

#### Models and Results:

I use k-NN, Decision Tree, SVM and Logistic Regression methods to build the models.

| Methods | Accuracy |
| ------- | -------- |
|k-NN|0.7825|
|Decision Tree|0.7200|
|SVM|0.7996|
|Logistic Regression|0.8131|

In this project, logistic regression is the best classifier model with highest predictive accuracy (81.31%).

