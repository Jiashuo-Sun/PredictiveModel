# Predictive Model

This is the Data Science Project at [SharpestMinds](https://www.sharpestminds.com/).

### Project Aims:

- Predict customer's behavior to retain them.
- Compare different methods and find the best one.
- Work on an on-hand data science project end-to-end.

### Data Source:

I got dataset from [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn). 

After reading [Hands-on: Predict Customer Churn](https://towardsdatascience.com/hands-on-predict-customer-churn-5c2a42806266), I created my own predictive models for customer churn.

### Data Preprocessing:

1. Replace binary variables with [0,1].
2. Change categorical data into numerical data. Use "dummies" function in Pandas.
3. Drop missing data. Some "TotalCharges" are blank spaces, replace the blank spaces with NaN and drop these samples. I did not replace them with zero because in this case, missing value in total charges does not mean that the charges are zero. Dropping them were more reasonable.
4. Split data into training set and testing set (80% / 20%).
5. Store the above two datasets. For small sets and **consistent experimentation**. This step is not necessary for large datasets or other projects, I did this for comparing the results of different methods on the same test set. Regularly, just need to test the model is functioning on test set with random spliting.

### Models and Results:

I used k-NN, Decision Tree, SVM and Logistic Regression methods to build the models.

| Methods | Accuracy |
| ------- | -------- |
|k-NN|0.7825|
|Decision Tree|0.7200|
|SVM|0.7996|
|Logistic Regression|0.8131|

In this project, logistic regression was the best classifier model with highest predictive accuracy (81.31%).

### Future Work:

All four methods I used are just simple classifiers, so other complicated methods (random forests, neural networks, etc) may have better performance based on this dataset. (When data size is not extremely large, algorithms will have more influence on the performance. So choose wisely~) 

Run some dimension reduction methods before modeling might be another approach, since I used all features in the dataset to complete the prediction. It might reduce the dataset size and will be suitable for larger dataset. 

**Update:**

After 6 months I finished this project, now it seems that there are plenty loopholes in this project: dataset was not preprocessed thoroughly, no feature selection was a huge flaw, and methods I used were simple. 
Despite so many flaws, this project provides the base pipeline of a data science project. Modeling may be fancy but data cleansing and feature engineering are more important for a project's success.

