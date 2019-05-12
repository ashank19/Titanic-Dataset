# Table of Contents

# Introduction
# Assessing
# Cleaning
# Feature Engineering
# Exploratory Analysis
# Regression Analysis
# Conclusions

# Importing relevant libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
%matplotlib inline

# Loading the data set into a pandas dataframe

gender=pd.read_csv('gender_submission.csv')
test=pd.read_csv('test.csv')
train=pd.read_csv('train.csv')

# Assessing the data

gender.info()

train.info()

train.describe(include='all')

# Assessing the first few columns.

train.head()

# Dataset Issues

# PassengerId column is of int type.

# Cabin information missing for most of the passengers.

# For the age column also dataset is missing which can prove to be a deciding factor in predicting survival.

# Two passengers having no information of Embarked ports.

# Cleaning the Dataset

# Define

# PasseingerId column is of int type rather it should be of str type.

# Code

train['PassengerId']=train['PassengerId'].astype(str)

# Test

train.info()

# Define

# The passengers having no information for embarked ports can be dropped as they are only two in count so it won't affect our analysis much.

# Code

train=train.drop(train[pd.isnull(train['Embarked'])].index)

# Test

train.info()

# Define

# As the cabin column is mostly empty so dropping the column.

# Code

train=train.drop(['Cabin'],axis=1)

# Test

train.info()

# Define

# The Age column being empty also from descriptive it is clear that about 75 persent of the passenger have age 38 so replacing the null values with mean of the entire column.

# Code

train['Age']=train['Age'].fillna(value=train.Age.mean())

# Test

train.info()

# Feature Engineering

# Here SibSp as stated above gives the no of sibling / spouses of the respective passenger aboard the titanic. So here segregating them into two columns as siblings and spouses with the assumption that spouses is one only for those above 18 years of age.

# Similarly for Parch segregating them into parents and children with the condition that parents at max can be two and rest are children.

# The task can be completed either using loops or functions. Functions have been used here to reduce the execution time.

train=train.reset_index(drop=True)

# Function for siblings and spouse column.

def Sibsp(df,Age,sib,i):
    if(sib != 0 and Age > 18):
        df.loc[i,'Sibling']=df.loc[i,'SibSp']-1
        df.loc[i,'Spouse']=df.loc[i,'SibSp']-df.loc[i,'Sibling']
    elif(sib != 0 and Age < 18):
        df.loc[i,'Sibling']=df.loc[i,'SibSp']
        df.loc[i,'Spouse']=0
    else:
        df.loc[i,'Spouse']=0
        df.loc[i,'Sibling']=0

# Function for parents and children column.

def ParCh(df,Par,i):
    if(Par >= 2):
        df.loc[i,'Children']=df.loc[i,'Parch']-2
        df.loc[i,'Parents']=df.loc[i,'Parch']-df.loc[i,'Children']
    else:
        df.loc[i,'Children']=0
        df.loc[i,'Parents']=df.loc[i,'Parch']

# Conversion into the required format.

for i in range(train.shape[0]):
    Sibsp(train,train.loc[i,'Age'],train.loc[i,'SibSp'],i)
    ParCh(train,train.loc[i,'Parch'],i)

# Testing the dataset for the feature Engineering performed above.

train

# Dropping the SibSp and Parch column.

train=train.drop(['SibSp','Parch'],axis=1)

# Checking for results

train.info()

# Converting the the siblings, parents, children and spouse into int type as these cannot be float type.

train['Sibling']=train['Sibling'].astype(int)
train['Spouse']=train['Spouse'].astype(int)
train['Parents']=train['Parents'].astype(int)
train['Children']=train['Children'].astype(int)

# Cleaning of testing dataset

test.info()

test=test.drop(['Cabin'],axis=1)
test['PassengerId']=test['PassengerId'].astype(str)

for i in range(test.shape[0]):
    Sibsp(test,test.loc[i,'Age'],test.loc[i,'SibSp'],i)
    ParCh(test,test.loc[i,'Parch'],i)

test=test.drop(['SibSp','Parch'],axis=1)

test['Sibling']=test['Sibling'].astype(int)
test['Spouse']=test['Spouse'].astype(int)
test['Parents']=test['Parents'].astype(int)
test['Children']=test['Children'].astype(int)

test['Age']=test['Age'].fillna(value=test.Age.mean())

    
