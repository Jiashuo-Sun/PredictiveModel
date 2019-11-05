import numpy as np
import pandas as pd

# read data from source file
data = pd.read_csv('.\data\WA_Fn-UseC_-Telco-Customer-Churn.csv')

# change binary data with value 0,1
data['gender'].replace(to_replace = ['Male','Female'],value = [1,0], inplace = True)
data['Partner'].replace(to_replace = ['Yes','No'], value = [1,0], inplace = True)
data['Dependents'].replace(to_replace = ['Yes','No'], value = [1,0], inplace = True)
data['PhoneService'].replace(to_replace = ['Yes','No'], value = [1,0], inplace = True)
data['PaperlessBilling'].replace(to_replace = ['Yes','No'], value = [1,0], inplace = True)
data['Churn'].replace(to_replace = ['Yes','No'], value = [1,0], inplace = True)

# dummy the category data and drop the original ones
temp1 = pd.get_dummies(data['MultipleLines'], prefix = 'MultipleLines')
temp2 = pd.get_dummies(data['InternetService'], prefix = 'InternetService')
temp3 = pd.get_dummies(data['OnlineSecurity'], prefix = 'OnlineSecurity')
temp4 = pd.get_dummies(data['OnlineBackup'], prefix = 'OnlineBackup')
temp5 = pd.get_dummies(data['DeviceProtection'], prefix = 'DeviceProtection')
temp6 = pd.get_dummies(data['TechSupport'], prefix = 'TechSupport')
temp7 = pd.get_dummies(data['StreamingTV'], prefix = 'StreamingTV')
temp8 = pd.get_dummies(data['StreamingMovies'], prefix = 'StreamingMovies')
temp9 = pd.get_dummies(data['Contract'], prefix = 'Contract')
temp10 = pd.get_dummies(data['PaymentMethod'], prefix = 'PaymentMethod')

data = pd.concat([data,temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8,temp9,temp10], axis = 1)
data = data.drop(['MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaymentMethod'], axis = 1)

# drop NaN in TotalCharges
data['TotalCharges'] = data['TotalCharges'].replace(" ",np.nan)
data.dropna(inplace = True)
data['TotalCharges'] = data['TotalCharges'].astype('float')
print(data.dtypes)

# split the whole data set into train/test (80%/20%)
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size = 0.2, random_state = 19)

# write train data and test data into new .csv file
train.to_csv(r'data\train_data.csv', index = False, header = True)
test.to_csv(r'data\test_data.csv', index = False, header = True)
