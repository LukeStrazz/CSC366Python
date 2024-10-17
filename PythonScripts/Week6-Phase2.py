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
Y_train = y[perm[:split_point].ravel()]     # For y value we only need the first column (only column that exists)
print('\nTraining shape:  \n' + str(X_train.shape) + '\ntraining targets shape: \n' + str(Y_train.shape))

X_test = x[perm[split_point:].ravel(),:]
Y_test = x[perm[split_point:].ravel()]
print('\nTesting shape:  \n' + str(X_test.shape) + ' \ntesting targets shape: \n' + str(Y_test.shape))

knn = neighbors.KNeighborsClassifier(53) # Best number of neighbors to use is square root of sample size. Here, 57
knn.fit(X_train, Y_train)
yhat = knn.predict(X_train)

print('Classification Accuracy: ', metrics.accuracy_score(yhat, Y_train))
print('Confusion Matrix:\n' + str(metrics.confusion_matrix(Y_train, yhat)))

#Automation of splitting data
PRC = 0.8
acc = np.zeros((10, ))
for i in range(10):
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = PRC)
    knn = neighbors.KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_train, Y_train)
    yhat = knn.predict(X_test)
    acc[i] = metrics.accuracy_score(yhat, Y_test)
print(acc)
acc.shape = (1,10)
print('Mean expected error ' + str(np.mean(acc[0])))
