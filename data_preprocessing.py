# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 21:07:37 2020
@author: parsis
"""
# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the Datasets
df = pd.read_csv('Data.csv')
print (df)
print ()
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3]) # Fit the imputer on x.
x[:, 1:3] = imputer.transform(x[:, 1:3]) # Fit to data, then transform it.

# Encoding the categorical data

# LabelEncoder
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x[:,0] = labelencoder_x.fit_transform(x[:,0])

# OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(handle_unknown='ignore')
x_df = onehotencoder.fit_transform(x[:,0].reshape(-1,1)).toarray()
x = np.concatenate((x_df,x[:,1:3]),axis=1)
print ("Shape of the array is: ",x.shape)
print ("Predictor variables:   \n",x)

# Another method OneHotEncoder
# from sklearn.compose import ColumnTransformer
# ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# x = np.array(ct.fit_transform(x), dtype=np.float)
# x

# For Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
print ("Dependent Variable:   ",y)

# Splitting the dataset into training and testing set
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=0)
print ("Training Size of x: ",xtrain.shape)
print ("Training Size of x: ",ytrain.shape)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
xtrain = sc_X.fit_transform(xtrain)
xtest = sc_X.transform(xtest)

sc_y = StandardScaler()
ytrain = sc_y.fit_transform(ytrain.reshape(-1,1))