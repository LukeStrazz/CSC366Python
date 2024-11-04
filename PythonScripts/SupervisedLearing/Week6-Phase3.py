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

PRC = 0.1
acc_r = np.zeros((10,4))
for i in range(10):
    X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=PRC)
    nn1 = neighbors.KNeighborsClassifier(n_neighbors=1)
    nn2 = neighbors.KNeighborsClassifier(n_neighbors=2)
    nn3 = neighbors.KNeighborsClassifier(n_neighbors=3)
    svc = svm.SVC()
    dt = tree.DecisionTreeClassifier()
    
    nn1.fit(X_train, Y_train)
    nn3.fit(X_train, Y_train)
    svc.fit(X_train, Y_train)
    dt.fit(X_train, Y_train)
    
    # Use model to predict:
    yhat_nn1 = nn1.predict(X_test)
    yhat_nn3 = nn3.predict(X_test)
    yhat_svc = svc.predict(X_test)
    yhat_dt = dt.predict(X_test)
    
    acc_r[i][0] = metrics.accuracy_score(yhat_nn1, Y_test)
    acc_r[i][1] = metrics.accuracy_score(yhat_nn3, Y_test)
    acc_r[i][2] = metrics.accuracy_score(yhat_svc, Y_test)
    acc_r[i][3] = metrics.accuracy_score(yhat_dt, Y_test)
print('Accuracy Matrix: \n' + str(acc_r) + '\n')
plt.boxplot(acc_r)
for i in range(4):
    xderiv = (i+1)*np.ones(acc_r[:,1].shape) + (np.random.rand(10,)-0.5)*0.1
    plt.plot(xderiv, acc_r[:,i],'ro',alpha=0.3)
ax = plt.gca() # Print plot
ax.set_xticklabels(['1-NN', '3-NN', 'SVM', 'Decision Tree'])
plt.ylabel('Accuracy')
plt.show()
plt.savefig('error_ms_accuracy.png', dpi=300, bbox_inches='tight')
    