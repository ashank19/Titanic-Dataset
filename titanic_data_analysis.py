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

# Exploring the dataset

bins=np.arange(0,train.Age.max()+10,10)
plt.hist(train['Age'],rwidth=0.6,bins=bins)
plt.title('Distribution of the dataset according to Age.')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show();

# From the above histogram it is clear that majority of the dataset contains people of age between 20 to 30, followed by those between 30 and 40.

sb.countplot(data=train,x='Survived',hue='Sex');

# The bar chart above compares the passengers survived or not on gender basis.

# From above it is clear that among those who survived females are high in number as compared to males and among who did not survive males are high in number.

sb.countplot(data=train,x='Survived',hue='Pclass');

# The bar chart above compares the passengers survived or not on socio economic status.

# The above chart shows that first class passengers were preferred over other passsengers at the time of rescue,which is quite evident from above graph as the number of third class passengers are high among those who did not survive.

plt.figure(figsize=[12,7])
sb.violinplot(data=train,x='Pclass',y='Fare')
plt.show();


# The violinplot above shows the distribution of fares of passengers depending upon their socio-economic status (Pclass).

# From the plot it is clear that passengers belonging to first class have the highest fares along with presence of outliers beyond 500. Those belonging to second class have their fares between 0 to 100.

# And those belonging to third class majority of their fares are below 50 and the distribution of fare is unimodal.

# The dot at the center of the violin plot depicts the median of the distribution thereby depicting that median among different class can be compared as

# (Pclass=1)median > (Pclass=1)median > (Pclass=1)median

# Converting the Sex column to a dumy variable

test

test[['Male','Gender']]=pd.get_dummies(test['Sex'])

test.drop(['Sex'],axis=1,inplace=True)

test

# As it is evident from above that if Gender is one it denotes a male or else a female thereby dropping male column so as to avoid redundancy in the matrix calculation.

test.drop(['Male'],axis=1,inplace=True)
test.info()

# Regression Analysis

# Dummies for gender in training dataset

train[['Male','Gender']]=pd.get_dummies(train['Sex'])
train.drop(['Sex'],axis=1,inplace=True)
train.drop(['Male'],axis=1,inplace=True)

# Importing relevant libraries

from sklearn.linear_model import LogisticRegression as L
x=train[['Gender','Age','Pclass']]
y=train['Survived']
reg=L()
reg.fit(x,y)

# Accuracy of the model

reg.score(x,y)

# Coefficients of features

reg.coef_

# Intercept

reg.intercept_

# Using F regression so as to ensure that the features used are significant.

from sklearn.feature_selection import f_regression

f_regression(x,y)

p_values=f_regression(x,y)[1]

p_values.round(3)

# From the p-values above it is clear that all the features are significant.

test.columns

pred=test.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Embarked',
       'Spouse', 'Sibling', 'Children', 'Parents'],axis=1)

y_pred=reg.predict(pred)

y_true=gender['Survived']

# Checking the Accuracy of the Model

from sklearn import metrics

metrics.accuracy_score(y_true,y_pred)

0.7392344497607656

# The accuracy of the model turns out to be 74% approximately on the testing dataset which is smaller than that of the training model.

# Variance Inflation Factor

from statsmodels.stats.outliers_influence import variance_inflation_factor

variables = train[['Gender','Age','Pclass']]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["features"] = variables.columns

# Checking the table

vif

# From above table it is clear that Vif for each of the features is normal except that only Pclass has vif close to 4.

# Regression taking Embarked port into consideration

train.Embarked.value_counts()
train[['C','Q','S']]=pd.get_dummies(train['Embarked'])
train

x=train[['Gender','Age','Pclass','Q','S']]
y=train['Survived']

reg=L()

reg.fit(x,y)

reg.score(x,y)

test.Embarked.value_counts()

test[['C','Q','S']]=pd.get_dummies(test['Embarked'])

test

pred=test.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Embarked',
       'Spouse', 'Sibling', 'Children', 'Parents','C'],axis=1)

y_pred=reg.predict(pred)

metrics.accuracy_score(y_true,y_pred)

# It is evident from above that if we take embarked port into consideration then then there is also slight reduction in the accuracy of the model. Thereby we would prefer dropping the embarked port feature according to this model.

# Conclusion

# The histogram for age distribution shows that majority of the dataset contains people of age between 20 to 30, followed by those between 30 and 40.

# First class passengers were preferred over other passsengers at the time of rescue,as the number of third class passengers are high among those who did not survive.

# The violinplot shows that distribution of fares of passengers depending upon their socio-economic status (Pclass).From the plot it is clear that passengers belonging to first class have the highest fares along with presence of outliers beyond 500. Those belonging to second class have their fares between 0 to 100.And those belonging to third class majority of their fares are below 50 and the distribution of fare is unimodal. Thus it can be concluded that fares for passengers of First class are higher than that of other class also they belong to the upper class of society.

# In the above regression analysis firstly regression has been performed on three features namely gender,age and passenger class where the accuracy of training and testing set were 79.6% and 73.9% respectively.

# In the second phase of regression analysis we decided to included embarked port also in the features there the accuracy of testing and training set were 79% and 72.24% respectively. So it can be concluded that included embarked port in our features does not increase the accuracy of our model so we consider dropping it.
       
