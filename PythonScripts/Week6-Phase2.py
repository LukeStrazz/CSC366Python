import pickle 
from sklearn import neighbors
from sklearn import datasets
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from sklearn import metrics

knn = neighbors.KNeighborsClassifier(n_neighbors=11) #Optimal KNN size is # of neighbors. Square root of sample size. Here, sqrt(4140) = 64 neighbors. Optimum number of neighbors we should use. 
with open('../DataFiles/dataset_small.pkl', 'rb') as file: #If this is a file we can open, procees (Meaning no restrictions). 'rb' means read binary
    data = pickle.load(file, encoding='latin1')
x, y = data

perm = np.random.permutation(y.size) # Permutation

print("Perm: ")
print(perm)

PRC = 0.8 # Precision 
split_point = int(np.ceil(y.shape[0]*PRC))
print("\nSplit point: ")
print(split_point)

X_train = x[perm[:split_point].ravel(),:]   # Get all the subset and columns for indecies (Get all columnms from dataset)
print("\nX_train: ")
print(X_train)

Y_train = y[perm[:split_point].ravel()]     # For y value we only need the first column (only column that exists)
print("\nY_train: ")
print(Y_train)

X_test = x[perm[split_point:].ravel(),:]
print("\nX_test: ")
print(X_test)

Y_test = x[perm[split_point:].ravel()]
print("\nY_test: ")
print(Y_test)