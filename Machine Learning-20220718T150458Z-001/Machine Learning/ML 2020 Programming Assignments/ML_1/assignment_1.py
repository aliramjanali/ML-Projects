#!/usr/bin/env python
# coding: utf-8

# In[244]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import array
from numpy.linalg import inv
get_ipython().run_line_magic('matplotlib', 'inline')


# # Familiarization with DataFrames, Matplotlib, Sklearn

# In[245]:


#CREATING DATAFRAME OF PLAYERS WITH THEIR SCORES IN 20 INNINGS
df = pd.DataFrame({   
    'P_1' :     [63, 46, 47, 50, 45, 71, 21, 81, 43,  4, 71, 55, 49, 55,  8, 33, 24,
       99, 16, 98],
    'P_2':[60, 21, 90, 41, 76, 60, 53, 10, 99,  6, 76, 35, 92, 35, 88, 56, 18,
       96, 81, 67],
    'P_3':[65, 80, 16, 75, 84, 28,  4, 27, 29, 48, 61, 64, 84, 31, 99,  6, 33,
        7, 60, 80],
    'P_4':[18, 18, 20, 77, 89, 76, 55,  0, 78, 95, 87, 27, 11, 26, 84, 23, 95,
       79,  0, 91],
    'P_5':[42, 90, 67, 21, 83, 10, 59, 89, 19, 71, 15, 56, 18, 69, 15, 68, 44,
       98, 31, 60],
    'P_6':[ 0,  9, 94, 91, 60, 23, 69, 40, 98, 46, 75, 12, 99, 65, 95, 39, 66,
       75, 40, 89],
    'P_7':[58, 69, 25, 13, 75, 75, 81, 78, 66, 45, 62, 25, 68,  1, 59, 50,  7,
       71, 75, 89],
    'P_8':[24, 71, 20, 56, 72,  9, 51, 15, 75, 29, 37, 73, 33,  7, 83, 21, 55,
       64, 11, 16],
    'P_9':[22, 26, 17, 57, 40, 62, 26, 22, 23,  5,  0, 46, 17, 36, 28, 41,  1,
       85, 34, 31],
    'P_10':[90,  1, 66, 96, 53, 87, 16, 19, 30, 18, 26, 67, 91, 18, 78, 53, 48,
       12, 28, 73],
    'P_11':[92, 82, 19, 36, 36, 97, 12, 62, 97, 34, 31, 73,  9, 23, 64, 96, 15,
        3,  3,  0],
    'P_12':[37, 79, 42, 20,  8, 53, 36, 37, 70, 96, 37, 52, 38, 55, 21, 93, 45,
       96, 48, 66]
})
df1=df.copy()
df2=df.copy()
df


# # Print a chosen record (row) or a chosen column (data field).

# In[246]:


df['P_1']#CHOSEN COLUMN


# # SELECTING ALL RECORDS WHERE SCORE IS GREATER THAN 50

# # I have shown only for 3 players like this we can select for other players also

# In[247]:


#SELECTING ALL RECORDS WHERE SCORE IS GREATER THAN 50
columns=['P_1', 'P_2', 'P_3','P_4','P_5','P_6','P_7','P_8','P_9','P_10','P_11','P_12']
df.loc[df['P_1']>50]


# In[248]:


#FOR P_2 COLUMN
df[df['P_2']>50]


# In[249]:


#FOR P_3 COLUMN 
df[df['P_3']>50]


# # SORTING ALL SCORES

# In[250]:


#SORTING ALL SCORES
df1 = df1.apply(lambda x: x.sort_values().values)
df1


# # ADDING EXTRA COLUMN SUM

# In[251]:


#ADDING EXTRA COLUMN SUM
df2["sum"] =df2.sum(axis=1)
df2


# # #FINDING THE PERCENTAGE OF EACH SCORE OF PLAYER IN EACH INNING
# #VECTORISED OPERATION

# In[252]:


#FINDING THE PERCENTAGE OF EACH SCORE OF PLAYER IN EACH INNING
#VECTORISED OPERATION
data1 = np.array(df)
vec = np.array(df2['sum'])
g=np.divide(data1.T,vec).T
k=g*100
datas = pd.DataFrame({'P_1': k[:, 0], 'P_2': k[:, 1],'P_3': k[:, 2],'P_4':k[:, 3],'P_5': k[:, 4],'P_6': k[:, 5],
                      'P_7': k[:, 6],'P_8': k[:, 7],'P_9': k[:, 8],'P_10': k[:, 9],'P_11': k[:, 10],'P_12': k[:,11]})
datas 


# # Use matplotlib library to make a scatter plot of columns that contain numeric data. Provide labels to the axes
# 
# 

# In[253]:


#RANDOM DATAFRAME TO PLOT SCATTER PLOT
data = np.random.randint(0,100,size=(200,2))
df3 = pd.DataFrame(data, columns=['A', 'B'])
df3


# In[254]:


df3.plot.scatter(x='A',
                     y='B',
                     c='DarkBlue')


# # SAVING AND READING PICKLE FILE

# In[255]:


#SAVING TO PICKLE FILE
df.to_pickle('df.pkl')


# In[256]:


#READING FROM PICKLE FILE
df4=pd.read_pickle('df.pkl')


# In[257]:


df4.head()


# #  Implementing Linear Regression

# In[323]:


#GENERATING RANDOM DATAFRAME TO IMPLEMENT LINEAR REGRESSION
data = np.random.randint(1,30,size=(100,2))
df4= pd.DataFrame(data, columns=['A','B'])
#A-->predictor
#B-->target
q=array(df4.values.tolist())


# In[324]:


k = np.shape(q)[0]
X = np.matrix([np.ones(k), q[:,0]]).T
y = np.matrix(q[:,1]).T


# In[325]:


#USING PSEUDO INVERSE
coeff=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)


# In[326]:


#INTERCEPT OF LINE
coeff[0]


# In[327]:


#SLOPE OF LINE
coeff[1]


# In[328]:


x1 = np.linspace(0, 30, 2)
y1 = np.array(coeff[0] + coeff[1] * x1)


# In[329]:


plt.figure(1)
plt.plot(x1, y1.T, color='red')
plt.scatter(q[:,0], q[:,1])
plt.xlabel("Values of A",weight = 'bold')
plt.ylabel("Values of B",weight = 'bold')
plt.show()


# # verification

# In[330]:


#VERIFICATION 
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X,y)


# In[331]:


#SLOPE OF LINE
model.coef_


# In[332]:


model.intercept_

