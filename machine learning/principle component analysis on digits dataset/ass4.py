#!/usr/bin/env python
# coding: utf-8

# In[10]:


from sklearn.datasets import load_digits
import numpy 
import pandas 

digits = load_digits()

myData = pandas.DataFrame(data = digits.data)

display(myData)


# In[11]:


# normalization using Z score
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(myData)

display(X_scaled)

# transpose of X scaled
display(X_scaled.T)


# In[12]:


# find covariance matrix
features = X_scaled.T
cov_matrix = numpy.cov(features)

display(cov_matrix)


# In[13]:


# find eigen values and eigen vectors
values, vectors = numpy.linalg.eig(cov_matrix)

display(values)

display(vectors)


# In[14]:


# find explained variance to check which top k features to choose

from matplotlib import pyplot as plt

explained_variances = []
for i in range(len(values)):
    explained_variances.append(values[i] / numpy.sum(values))
    
display(explained_variances)

# plt.figure(figsize=(6, 4))
plt.bar(range(len(explained_variances)), explained_variances,align='center',label='individual explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')


# In[15]:


# as you can see in the above graph - top ten features have most of the significance


# making reduced data set

# projected Value is X normalized matrix . eigen vector

projectedList = []
colsNamesList = []

projected = X_scaled.dot(vectors.T[0])
reduced = pandas.DataFrame(projected, columns = ['PC1'])

for i in range(1 , 10):
    reduced["PC{}".format(i+1)] = X_scaled.dot(vectors.T[i])

reduced['Y'] = digits.target

display(reduced)


# In[16]:


# using in built PCA function
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
X_new = pca.fit_transform(X_scaled)

display(X_new)


# In[ ]:




