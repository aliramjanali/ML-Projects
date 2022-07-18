#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
# Load dataset
#iris dataset is taken
data = datasets.load_iris()
X = data.data[:, :2]
#initialize K
K=12
#selected k=10 nd 30 for analysing
#in_cent = np.array([[3, 3], [6, 2], [8, 5]])
in_cent = np.array([[1, 2], [6, 4], [3, 9]])


# # Implement K-means(I have tried for k=3 here) for more higher k results are mentioned in report

# In[28]:


"""
   It Returns the closest centroids in array
   (each row=example). array= each entry in range [1..K]
   
       centroids: array(K, n)
   Returns:
         array(training examples, 1)
   """

def ClosestCentroids(X, cent):
  
   K = cent.shape[0]

   # Initialise index.
   ind = np.zeros((X.shape[0], 1), dtype=np.int8)

   # move on every example, find closest centroid, and store
   # index inside index at the suitable location. 
   
   for i in range(X.shape[0]):
       dist = np.linalg.norm(X[i] - cent, axis=1)
       mindst = np.argmin(dist)
       ind[i] = mindst
   
   return ind


#closest centroids for examples.

ind =ClosestCentroids(X, in_cent)

print('The nearest centroids for first 3 examples are: \n')
print(ind[:3])


# In[29]:


"""
The given Returns the newly centroids by formulating data point means for each centroid. 

Returns:
    centroids: array(of centroids, 2)
"""

def computeCentroids(X, ind, K):

 a, b = X.shape

# Initialize centroid(cent) matrix.

 cent = np.zeros((K, b))

# Move over all the centroid and calculate mean of all points 

 for i in range(K):
    cent[i, :] = np.mean(X[ind.ravel() == i, :], axis=0)

 return cent

# Find means based on the nearest centroids from the previous part.

cent = computeCentroids(X, ind, K)

print('Centroids calculated after finding of initial nearest centroids: \n')

print(cent)


# In[30]:


def plotPoints(X, ind, K):
    
   # Plots data points in X

    # Create a colors list.
    col = [plt.cm.tab20(float(j) / 10) for j in ind]

    
    plt.scatter(X[:,0], X[:,1], c=col, alpha=0.6, s=3)

# function to display the progress of K-Means.

   
    """each data point centroid is plotted with colors assigned to it.
    A line is also plotted between current and prev locations of the centroid"""

   

def ProgresskMeans(X, cent, prev, ind, K, i):
 
    # Plot  example.
    plotPoints(X, ind, K)

    # Plot the centroids as black x's.
    
    plt.scatter(cent[:,0], cent[:,1],
                marker='x', c='k', s=101, linewidth=2)

  
    for i in range(cent.shape[0]):
        plt.plot([cent[i, :][0], prev[i, :][0]],
                 [cent[i, :][1], prev[i, :][1]], c='k')
    
    plt.title('No of iteration {:d}'.format(i+1))

# Create a function to run the K-means algorithm.
 
    """
    K-Means algorithm runs on matrix X with input as in_cent .
    runkMeans returns 
    centroids, a K x n matrix of the computed centroids and id, a m x 1 
    vector of centroid assignments
    
    """

def kMeans(X, in_cent, m_iter, plot_progress):
   
    # Initialization
    
    a, b = X.shape
    K = in_cent.shape[0]
    cent = in_cent
    prev_cent = cent
    ind = np.zeros((a, 1))
    
    # Running of K-Means.
  
    plt.ion()
    for i in range(m_iter):
       
        print('No of K-Means iteration {}/{}...'.format(i, max_iters))
        
      
        ind=ClosestCentroids(X, cent)
        
        
        if plot_progress:
            ProgresskMeans(X, cent, prev_cent, ind, K, i)
            prev_cent = cent

        # compute new centroids.
        
        cent = computeCentroids(X, ind, K)

    return cent, ind


K = 3
m_iter = 10

# here we set centroids to specific values

#in_cent = np.array([[1, 2], [6, 4], [3, 9]])

# Running K-Means algorithm.

cent, ind = kMeans(X, in_cent, m_iter, plot_progress=True)
print('\nK-Means completed.')


# # # Implement PCA(recovered data is written in X_r)

# In[18]:


from scipy.io import loadmat
#dataset of images 
data = loadmat('ex7faces.mat')
X = data["X"]

print("X shape: ", X.shape)
plt.figure(figsize=(7, 7))
plt.scatter(X[:,0], X[:,1], edgecolors='b', facecolors='none')
plt.title("Fig1:Dataset")
plt.show()

 
"""n rows are selected from X,
and plotted as (length of image vector)^2 
grayscale pixel images, and one final figure is created."""

    

def display(X, n):
   
    # removing the unnecessary space between the subplots with gridspec.
    
    figu, axar = plt.subplots(n, n,
                             figsize=(7, 7),
                             gridspec_kw={'wspace':0,
                                          'hspace':0})
    ind = 0
    for i in range(n):
        for j in range(n):            
            
            pix = X[ind] 
            pix = pix.reshape(-1, 32) # shape(32, 32)
            axar[i,j].imshow(pix.T, cmap='gray')
            # Removing ticks.
            axar[i,j].set_xticks([])
            axar[i,j].set_yticks([])
            
            axar[i,j].axis('off')
            ind += 1
    plt.show()

display(X, 5)


# In[22]:


import scipy.linalg as linalg

# function to normalize features.

def Normalize(X):
   
    mui = np.mean(X, axis=0)
    X_n = X - mui
    
    sigmaa = np.std(X_n, axis=0, ddof=1)
    X_n = X_n / sigmaa
    
    return X_n, mui, sigmaa

# Calculate the eigenvectors and eigenvalues.

def pca(X):
   
    # Initialization
    a, b = X.shape
    
    # Initialize P and Q.
    
    P = np.zeros(b)
    Q = np.zeros(b)
    
    # compute the covariance matrix by dividing the number of examples(a).
    
    sigmaa = (1. / a) * np.dot(X.T, X)
    
    # Calculate the eigenvectors and eigenvalues  of covariance matrix
    
    
    P, Q, V = linalg.svd(sigmaa)
    Q= linalg.diagsvd(Q, len(Q), len(Q))

    return P, Q


X_n, mui, _ =Normalize(X)

# Running PCA.

P, Q = pca(X_n)

# Plotting the eigenvectors centered at mean of data.

plt.figure(figsize=(7, 7))
plt.scatter(X[:,0], X[:,1], edgecolors='b', facecolors='none')
plt.title("Fig: eigenvectors of the dataset.")

# Compute the pairs of points to draw the lines.

p = mui
p1 = mui + 1.5 * Q[0,0] * P[:,0].T
p2 = mui + 1.5 * Q[1,1] * P[:,1].T
plt.plot([p[0], p1[0]], [p[1], p1[1]], c='k', linewidth=3)
plt.plot([p[0], p2[0]], [p[1], p2[1]], c='k', linewidth=3)
plt.show()

print('Top eigenvector:')
print('P[:,0]= {:f} {:f}'.format(P[0,0], P[1,0]))


# # Normalization, Reconstruction and projecting back

# In[23]:



#function to project normalized inputs X
# into dimensional space spanned by the first
# K columns of P

def pData(X, P, K):
    
    # Initialize Z.
    H = np.zeros((X.shape[0], K))
    
    # projection of the data is calculated using 
    # top K eigenvectors in P.
    
    P_red = P[:,:K]
    H= np.dot(X, P_red)

    return H

#function to recover the data.

"""This function recovers the approx. original data
which has been reduced to K dimensions.
 approximate reconstruction is returned in X_r.
    """
def rData(H, P, K):
   
    # Initialize X_r.
    
    X_r = np.zeros((H.shape[0], P.shape[0]))
    
    # approximation of the data is calculated by projecting back
    # onto the original space using the top K eigenvectors in U.
    
    P_red = P[:,: K]
    X_r = np.dot(H, P_red.T)

    return X_r

# Projecting the data in K = 1 dimension.
K = 1
H = pData(X_n, P, K)
print('1st example projection: %.3f'% H[0])


X_r = rData(H, P, K)
print('\n 1st example approximation: {0:.6f} {0:.3f}'.format(X_r[0,0], X_r[0,1]))


# plotting the normalized X dataset.

plt.figure(figsize=(7, 7))
plt.scatter(X_n[:,0], X_n[:,1], edgecolors='b', facecolors='none')
plt.title("Fig:data after PCA(normalized and projected).")


plt.scatter(X_r[:,0], X_r[:,1], edgecolors='r', facecolors='none')

for j in range(X_n.shape[0]):
    plt.plot([X_n[j,:][0], X_r[j,:][0]],
             [X_n[j,:][1], X_r[j,:][1]],
             linestyle='--', color='k', linewidth=2)
plt.show()


# In[ ]:




