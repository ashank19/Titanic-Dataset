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
