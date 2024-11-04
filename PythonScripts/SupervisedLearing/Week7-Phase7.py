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
from sklearn.model_selection import KFold
from sklearn import tree

knn = neighbors.KNeighborsClassifier(n_neighbors = 11) #Optimal KNN size is # of neighbors. Square root of sample size. Here, sqrt(4140) = 64 neighbors. Optimum number of neighbors we should use. 
with open('../../DataFiles/dataset_small.pkl', 'rb') as file: #If this is a file we can open, procees (Meaning no restrictions). 'rb' means read binary
    data = pickle.load(file, encoding='latin1')
x, y = data

kf = KFold(n_splits=10, shuffle=True, random_state=0)

C = np.arange(2,20,)
acc = np.zeros((10,18))
i=0
for train_index, val_index in kf.split(y):
    X_train, X_val = x[train_index], x[val_index]
    y_train, y_val = y[train_index], y[val_index]
    j=0
    for c in C:
        dt = tree.DecisionTreeClassifier(min_samples_leaf=1,max_depth=c)
        dt.fit(X_train, y_train)
        yhat = dt.predict(X_val)
        acc[i][j] = metrics.accuracy_score(yhat,y_val)
        j=j+1
plt.boxplot(acc)
for i in range(18):
    xdrive = (i+1) * np.ones(acc[:,i].shape) + (np.random.rand(10,)-0.5)*0.1
    plt.plot(xdrive, acc[:,i],'ro',alpha=0.3)
    
print('Mean Accuracy: ' + str(np.mean(acc,axis=0)))
print('Selected model index: ' + str(np.argmax(np.mean(acc, axis=0))))
print('Complexity: ' + str(C[np.argmax(np.mean(acc, axis=0))]))
plt.ylim(0.7, 1.0)
fig = plt.gcf()
fig.set_size_inches(12, 5)
plt.xlabel('Complexity')
plt.ylabel('Accuracy')
plt.show()