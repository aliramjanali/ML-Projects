# -*- coding: utf-8 -*-
"""2.2 Linear Regration using gradient descent.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dW9R9R4S5ZyTvefJBmFhqy42xUkzuDnl
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
dataset=pd.read_csv('/content/housing.csv')
dataset.head()


def costFun(x,y,theta,lamb):
    h=np.dot(theta,x.T)
    J=(1/(2*len(x)))*np.sum(np.square(h-y))+(lamb/(2*len(x)))*np.sum(np.square(theta[1:]))
    return J

#Gradient Descent Function
def gradient_decent(x,y,theta,alpha,total_iter,lamb):
    m=len(x)
    for i in range(0,total_iter):
        h=np.dot(theta,x.T)
        theta[0]-=(alpha/m)*np.sum((h-y)*x.iloc[:,0])
        for j in range(1,len(x.columns)):
            theta[j]-=((alpha/m)*(np.sum((h-y)*x.iloc[:,j])+(lamb)*theta[j]))
    return theta


#Preparing the data set for training
X=dataset.drop(columns=["ocean_proximity","median_house_value"])
Y=dataset["median_house_value"]

for i in range(0,len(X.columns)):
    X.iloc[:,i]=(X.iloc[:,i]-min(X.iloc[:,i]))/((max(X.iloc[:,i])-min(X.iloc[:,i])))
Y=(Y-min(Y))/(max(Y)-min(Y))
X = pd.concat([pd.Series(1, index = X.index, name = '00'), X], axis=1)

#predict Function
def predictHousePrice(x,alpha,theta,num_iter):
   
    predict_price=np.dot(theta,x.T)
    return predict_price

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=123)
alpha=.001
lamb=1
total_iter=5000

train_theta= gradient_decent(x_train,y_train,[.5]*len(X.columns),alpha,total_iter,lamb)
cost=costFun(x_test,y_test,train_theta,lamb)
print(cost)

y_pred=predictHousePrice(x_test,alpha,[.5]*len(X.columns),total_iter)  
print(y_pred)