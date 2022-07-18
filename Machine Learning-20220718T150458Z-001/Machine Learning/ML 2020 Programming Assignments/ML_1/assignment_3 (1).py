#!/usr/bin/env python
# coding: utf-8

# In[553]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[554]:

#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Linear Regression Example
=========================================================
The example below uses only the first feature of the `diabetes` dataset,
in order to illustrate the data points within the two-dimensional plot.
The straight line can be seen in the plot, showing how linear regression
attempts to draw a straight line that will best minimize the
residual sum of squares between the observed responses in the dataset,
and the responses predicted by the linear approximation.

The coefficients, residual sum of squares and the coefficient of
determination are also calculated.
"""
print(__doc__)


# Code source: Jaques Grobler
# License: BSD 3 clause


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

df=pd.read_csv("heart.csv")


# In[555]:


df.head()
#1 Target means person is having heart disease
#0 Target means person is not having heart disease


# In[556]:


df['target'].value_counts()
#Balanced dataset


# In[557]:


## DESCRIPTION
'''cp---chest pain type
trestbps->resting blood pressure (in mm Hg on admission to the hospital)
chol->serum cholestoral in mg/dl
fbs->(fasting blood sugar &gt; 120 mg/dl) (1 = true; 0 = false)
restecg->resting electrocardiographic results
thalach->maximum heart rate achieved
exang->exercise induced angina (1 = yes; 0 = no)
oldpeak->ST depression induced by exercise relative to rest
slope->the slope of the peak exercise ST segment
ca->number of major vessels (0-3) colored by flourosopy
thal->3 = normal; 6 = fixed defect; 7 = reversable defect
target->1 or 0'''


# # Perceptron algorithm result(selected columns 3 and 7 as input with target as label)

# In[558]:


X= df.iloc[:,[3,7]].values
y = df.iloc[:,13].values
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x = sc.fit_transform(X)
from sklearn.linear_model import Perceptron
perceptron = Perceptron(random_state = 0)
perceptron.fit(x, y)
predicted = perceptron.predict(x)
perceptron.score(x,y)
#0.6039603960396039


# In[559]:


from matplotlib.colors import ListedColormap
plt.clf()
X_set, y_set = x, y

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, perceptron.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('navajowhite', 'darkkhaki')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Perceptron Classifier (Decision boundary for Yes vs the No)')
plt.xlabel('trestbps')
plt.ylabel('thalach')
plt.legend()
plt.show()


# # train test split
# 

# # Half Space with splits: 80:20 (80% training, 20% testing)

# In[560]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df.drop('target',axis=1),df['target'],test_size=0.2,random_state=42)


# In[561]:


#Scaling
X_train=np.array(X_train)
y_train=np.array(y_train)
X_test=np.array(X_test)
y_test=np.array(y_test)
X_train = X_train.astype(float)
y_train = y_train.astype(float)
X_test = X_test.astype(float)
y_test = y_test.astype(float)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[562]:


from sklearn import metrics
from sklearn.linear_model import Perceptron
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(X_train,y_train)
y_pred =clf.predict(X_test)
metrics.accuracy_score(y_test, y_pred)
#0.8524590163934426


# # classification_report

# In[563]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# # AUC Curve

# In[564]:


y_pred_proba = clf._predict_proba_lr(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# 
# # Logistic Regression with splits: 80:20 (80% training, 20% testing)

# In[565]:


X_train,X_test,y_train,y_test=train_test_split(df.drop('target',axis=1),df['target'],test_size=0.2,random_state=42)


# In[566]:


X_train=np.array(X_train)
y_train=np.array(y_train)
X_test=np.array(X_test)
X_train = X_train.astype(float)
y_train = y_train.astype(float)
X_test = X_test.astype(float)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[567]:


from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
clf.fit(X_train,y_train)
y_pred =clf.predict(X_test)
metrics.accuracy_score(y_test, y_pred)
#0.8524590163934426


# # classification_report

# In[568]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


#  # AUC Curve

# In[569]:


y_pred_proba = clf.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# # SVM classifier (using a linear kernel) with splits: 80:20 (80% training, 20% testing)

# In[570]:


X_train,X_test,y_train,y_test=train_test_split(df.drop('target',axis=1),df['target'],test_size=0.2,random_state=42)


# In[571]:


X_train=np.array(X_train)
y_train=np.array(y_train)
X_test=np.array(X_test)
X_train = X_train.astype(float)
y_train = y_train.astype(float)
X_test = X_test.astype(float)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[572]:


clf=SVC(kernel='linear',probability=True)
clf.fit(X_train,y_train)
y_pred =clf.predict(X_test)
metrics.accuracy_score(y_test, y_pred)
#0.8688524590163934


# # classification_report

# In[573]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# #  AUC Curve

# In[574]:


y_pred_proba = clf.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# #  SVM classifier(Gaussian) with splits: 80:20 (80% training, 20% testing)

# In[575]:


X_train,X_test,y_train,y_test=train_test_split(df.drop('target',axis=1),df['target'],test_size=0.2,random_state=42)


# In[576]:


X_train=np.array(X_train)
y_train=np.array(y_train)
X_test=np.array(X_test)
X_train = X_train.astype(float)
y_train = y_train.astype(float)
X_test = X_test.astype(float)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[577]:


from sklearn.svm import SVC
clf= SVC(kernel="rbf", gamma="auto", C=1,probability=True)
clf.fit(X_train,y_train)
y_pred =clf.predict(X_test)
metrics.accuracy_score(y_test, y_pred)
#0.8688524590163934


# # AUC Curve

# In[578]:


import sklearn.metrics as metrics
y_pred_proba = clf.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# # classification_report

# In[579]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# # SVM classifier(Polynomial) with splits: 80:20 (80% training, 20% testing)

# In[580]:


X_train,X_test,y_train,y_test=train_test_split(df.drop('target',axis=1),df['target'],test_size=0.2,random_state=42)


# In[581]:


X_train=np.array(X_train)
y_train=np.array(y_train)
X_test=np.array(X_test)
X_train = X_train.astype(float)
y_train = y_train.astype(float)
X_test = X_test.astype(float)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[582]:


clf= SVC(kernel='poly',probability=True)
clf.fit(X_train,y_train)
y_pred =clf.predict(X_test)
metrics.accuracy_score(y_test, y_pred) 
#0.9016393442622951


# # classification_report

# In[583]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[584]:


y_pred_proba = clf.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# # Logistic Regression using the SGD procedure with splits: 80:20 (80% training, 20% testing)

# In[585]:


X_train,X_test,y_train,y_test=train_test_split(df.drop('target',axis=1),df['target'],test_size=0.2,random_state=0)


# In[586]:


X_train=np.array(X_train)
y_train=np.array(y_train)
X_test=np.array(X_test)
X_train = X_train.astype(float)
y_train = y_train.astype(float)
X_test = X_test.astype(float)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[587]:


from sklearn.linear_model import SGDClassifier
clf=SGDClassifier(loss='log',random_state=0)
clf.fit(X_train,y_train)
y_pred =clf.predict(X_test)
metrics.accuracy_score(y_test, y_pred)
#0.7868852459016393


# # classification_report

# In[588]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# # AUC Curve

# In[589]:


y_pred_proba = clf.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# # Soft SVM with Regularization and support vectors(selected columns 3 and 7 as input with target as label)

# # A) Calculating support vectors for (70:30,80:20,90:10) split  in sequence

# In[609]:


X_train,X_test,y_train,y_test=train_test_split(df.drop('target',axis=1),df['target'],test_size=0.3,random_state=0)


# In[610]:


from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X= X_train.iloc[:,[3,7]].values
y = y_train
x = sc.fit_transform(X)


# In[611]:


from sklearn.svm import SVC # "Support vector classifier"
model = SVC(kernel='linear', C=50)
model.fit(x, y)


# In[612]:


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


# In[613]:


plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model);


# In[614]:


len(model.support_vectors_)
#149


# In[615]:


X_train,X_test,y_train,y_test=train_test_split(df.drop('target',axis=1),df['target'],test_size=0.2,random_state=0)


# In[616]:


from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X= X_train.iloc[:,[3,7]].values
y = y_train
x = sc.fit_transform(X)


# In[617]:


from sklearn.svm import SVC # "Support vector classifier"
model = SVC(kernel='linear', C=50)
model.fit(x, y)


# In[618]:


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


# In[619]:


plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model);


# In[620]:


len(model.support_vectors_)
#172


# In[621]:


X_train,X_test,y_train,y_test=train_test_split(df.drop('target',axis=1),df['target'],test_size=0.1,random_state=0)


# In[622]:


from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X= X_train.iloc[:,[3,7]].values
y = y_train
x = sc.fit_transform(X)


# In[623]:


from sklearn.svm import SVC # "Support vector classifier"
model = SVC(kernel='linear', C=50)
model.fit(x, y)


# In[624]:


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


# In[625]:


plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model);


# In[626]:


len(model.support_vectors_)
#194


# # Effect of Regulrization parameter( C) on support vectors and performance with two C values  C=0.1 and C=20
# 

# In[627]:


X_train,X_test,y_train,y_test=train_test_split(df.drop('target',axis=1),df['target'],test_size=0.2,random_state=0)


# In[628]:


from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X= X_train.iloc[:,[3,7]].values
y = y_train
x = sc.fit_transform(X)


# In[629]:


from sklearn.svm import SVC # "Support vector classifier"
model = SVC(kernel='linear', C=0.1)
model.fit(x, y)


# In[630]:


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


# In[631]:


plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model);


# In[632]:


len(model.support_vectors_)
#182


# In[633]:


X_train,X_test,y_train,y_test=train_test_split(df.drop('target',axis=1),df['target'],test_size=0.2,random_state=0)


# In[634]:


from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X= X_train.iloc[:,[3,7]].values
y = y_train
x = sc.fit_transform(X)


# In[635]:


from sklearn.svm import SVC # "Support vector classifier"
model = SVC(kernel='linear', C=20)
model.fit(x, y)


# In[636]:


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


# In[637]:


plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model);


# In[638]:


len(model.support_vectors_)
#172

