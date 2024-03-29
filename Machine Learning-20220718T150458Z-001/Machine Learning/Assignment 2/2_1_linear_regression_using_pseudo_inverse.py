# -*- coding: utf-8 -*-
"""2.1 linear regression using Pseudo Inverse.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MqHQ2od0xA-Uh8Mv5xgEItwdXGj4lsrT
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# import seaborn
import seaborn as sns
# %matplotlib inline

data1 = pd.read_csv("/content/heart_failure_clinical_records_dataset.csv")
data1.head()
Y = data1.platelets.values
X = data1.creatinine_phosphokinase.values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

bplot= sns.scatterplot('creatinine_phosphokinase','platelets',data=data1)
bplot.axes.set_title("creatinine_phosphokinase vs platelets: Scatter Plot",fontsize=16)
bplot.set_ylabel("platelets", fontsize=16)
bplot.set_xlabel("creatinine_phosphokinase", fontsize=16)


X_mat=np.vstack((np.ones(len(X)), X)).T
X_mat[0:5,]

beta_hat = np.linalg.inv(X_mat.T.dot(X_mat)).dot(X_mat.T).dot(Y)

# predict using coefficients
yhat = X_mat.dot(beta_hat)


#Linear Regression Model Estimates
print(beta_hat)
# plot data and predictions
plt.scatter(X, Y)
plt.plot(X, yhat, color='red')


#Verifying Linear Regression Model Estimates using Scikit-learn
print()
print("Linear Regression Model Estimates using Scikit-learn")
regression = LinearRegression()
linear_model = regression.fit(X[:,np.newaxis],Y)
print(linear_model.intercept_)
print(linear_model.coef_)