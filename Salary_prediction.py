import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading the data

train= pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#let's check the data type

train.dtypes

train.describe()

#get a list of categorical variable

categorical_variable = train.dtypes.loc[train.dtypes=='object'].index
print(categorical_variable)

train[categorical_variable].apply(lambda x : len(x.unique()))

train['Race'].value_counts()
train['Race'].value_counts()/train.shape[0]

train['Native.Country'].value_counts()
train['Native.Country'].value_counts()/train.shape[0]

#print the cross-tabulation

ct= pd.crosstab(train['Sex'],train['Income.Group'], margins= True)
print(ct)

#we caN ALSO PLOT IT BY USING STACK CHART

#%matplotlib inline
ct.iloc[:-1,:-1].plot(kind='bar', stacked=True,color=['red','blue'],grid= False).plt

ct1 = pd.crosstab(train['Occupation'],train['Income.Group'],margins=True)
train.plt('Age','Hours.Per.Week', kind='Scatter')

train.boxplot(column="Hours.Per.Week",by="Sex")


#checking the missing value in train data

train.isnull().sum()

#we can also check the null value by apply and lamda combination

train.apply(lambda x: sum(x.isnull()))


#checking the misssing value in test data

test.isnull().sum

#by apply and lambda

test.apply(lambda x : sum(x.isnull()))

#import function

from scipy.stats import mode

#lets try to impute missing value

mode(train['Workclass']).mode[0]

#lets impute the missing value

var_to_impute = ['Workclass','Occupation','Native.Country']
for var in var_to_impute:
    train[var].fillna(mode(train[var]).mode[0],inplace=True)
    test[var].fillna(mode(train[var]).mode[0],inplace=True)