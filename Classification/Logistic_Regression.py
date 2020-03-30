# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 20:03:15 2020

@author: parsis
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
dataset.head()
print ("Size of dataset: ",dataset.shape)
x = dataset.iloc[:,[2,3]]
y = dataset.iloc[:,-1]
y
# Splitting the dataset into training set and testing set
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size = 0.2,random_state = 0)
print ("taining size of dataset: ",xtrain.shape)
print ("taining size of dataset: ",ytrain.shape)
print ("testing size of dataset: ",xtest.shape)
print ("testing size of dataset: ",ytest.shape)

# Check whether there is null data
print ("Total nos of null data\n")
print (dataset.isnull().sum())

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain)
xtest = sc_x.fit_transform(xtest)

# Fitting logitic regression to training set
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0)
model.fit(xtrain,ytrain)

# Predicting test set result
y_pred = model.predict(xtest)
print (y_pred)
#print (ytest)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix,r2_score
cm = confusion_matrix(ytest,y_pred)
print ("Confusion Metrics: \n",cm)
print ("R2 Score:            ",r2_score(ytest,y_pred))

# Visualizing the training set result
from matplotlib.colors import ListedColormap
xset, yset = xtrain, ytrain
# 0.01 resolution
x1,x2 = np.meshgrid(np.arange(start = xset[:,0].min()-1, stop = xset[:,0].max()+1,step = 0.01),
                    np.arange(start = xset[:,1].min()-1, stop = xset[:,1].max()+1,step = 0.01))
plt.contourf(x1,x2,model.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i, j in enumerate(np.unique(yset)):
    plt.scatter(xset[yset==j,0],xset[yset==j,1],
                c = ListedColormap(('red','green'))(i),label=j)
plt.title('LogisticRegression (Training Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Family')
plt.legend()
plt.show()

# Visualizing the testing set result
from matplotlib.colors import ListedColormap
xset, yset = xtest, ytest
x1,x2 = np.meshgrid(np.arange(start = xset[:,0].min()-1, stop = xset[:,0].max()+1,step = 0.01),
                    np.arange(start = xset[:,1].min()-1, stop = xset[:,1].max()+1,step = 0.01))
plt.contourf(x1,x2,model.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75,cmap=ListedColormap(('red','green')))
# Limits x and y
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i, j in enumerate(np.unique(yset)):
    plt.scatter(xset[yset==j,0],xset[yset==j,1],
                c = ListedColormap(('red','green'))(i),label=j)
plt.title('LogisticRegression (Testing Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Family')
plt.legend()
plt.show()