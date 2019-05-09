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
