import pandas as pd
import numpy as np


data = pd.read_csv(r'data/train_data.csv')

y = data['Churn'].values
x = data.drop(['customerID','Churn'],axis = 1)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 11)

from sklearn import preprocessing
x_train = preprocessing.StandardScaler().fit(x_train).transform(x_train)
x_test = preprocessing.StandardScaler().fit(x_test).transform(x_test)

# input test dataset

test_data = pd.read_csv(r'data/test_data.csv')
testx = test_data.drop(['customerID','Churn'],axis = 1)
testx = preprocessing.StandardScaler().fit(testx).transform(testx)

testy = test_data['Churn'].values

# k-NN classifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# find best k
Kmax = 20
acc_mean = np.zeros(Kmax)
for i in range(Kmax):
    knn = KNeighborsClassifier(n_neighbors = i + 1).fit(x_train,y_train)
    yhat = knn.predict(x_test)
    acc_mean[i] =  metrics.accuracy_score(y_test,yhat)

print('Best k is: ' + str(acc_mean.argmax()+1))
print('Best accuracy is: ' + str(acc_mean.max()))

# final knn classifier
knn = KNeighborsClassifier(n_neighbors = 14).fit(x_train, y_train)

testyhat = knn.predict(testx)
print("The k-NN's accuracy is "+str(metrics.accuracy_score(testyhat,testy)))

# Decision Tree

from sklearn.tree import DecisionTreeClassifier

DTree = DecisionTreeClassifier().fit(x_train,y_train)
yhat = DTree.predict(x_test)
metrics.accuracy_score(y_test,yhat)

testyhat = DTree.predict(testx)
print("The Decision Tree's accuracy is " + str(metrics.accuracy_score(testyhat,testy)))

# Support Vector Machine

from sklearn import svm

clf = svm.SVC(kernel = 'rbf').fit(x_train, y_train)
yhat = clf.predict(x_test)
metrics.accuracy_score(y_test,yhat)

testyhat = clf.predict(testx)
print("The SVM's accuracy is "+ str(metrics.accuracy_score(testyhat, testy)))

# Logistic Regression

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression().fit(x_train, y_train)
yhat = lr.predict(x_test)
metrics.accuracy_score(y_test,yhat)

testyhat = lr.predict(testx)
print("The Logistic Regression's acuracy is " + str(metrics.accuracy_score(testyhat, testy)))




