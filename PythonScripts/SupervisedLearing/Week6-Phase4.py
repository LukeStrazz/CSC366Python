import pickle
from sklearn import neighbors
from sklearn import datasets
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import tree 
from sklearn import svm
from sklearn import metrics

knn = neighbors.KNeighborsClassifier(n_neighbors=11) #Optimal KNN size is # of neighbors. Square root of sample size. Here, sqrt(4140) = 64 neighbors. Optimum number of neighbors we should use. 
with open('../../DataFiles/dataset_small.pkl', 'rb') as file: #If this is a file we can open, procees (Meaning no restrictions). 'rb' means read binary
    data = pickle.load(file, encoding='latin1')
x, y = data

MAXN = 700
fig = plt.figure()
fig.set_size_inches(6,5)
plt.plot(1.25*np.random.rand(MAXN, 1), 1.25*np.random.rand(MAXN,1),'r.',alpha=0.3)
plt.plot(8+1.5*np.random.rand(MAXN, 1), 5+1.5*np.random.rand(MAXN,1),'b.',alpha=0.3)
plt.plot(5+1.5*np.random.rand(MAXN, 1), 5+1.5*np.random.rand(MAXN,1),'g.',alpha=0.3)
plt.show()