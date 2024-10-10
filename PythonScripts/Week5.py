import pickle 
from sklearn import neighbors
from sklearn import datasets
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from sklearn import metrics

knn = neighbors.KNeighborsClassifier(n_neighbors=64) #Optimal KNN size is # of neighbors. Square root of sample size. Here, sqrt(4140) = 64 neighbors. Optimum number of neighbors we should use. 
with open('../DataFiles/dataset_small.pkl', 'rb') as file: #If this is a file we can open, procees (Meaning no restrictions). 'rb' means read binary
    data = pickle.load(file, encoding='latin1')
x, y = data
x_shape = None
y_shape = None

if hasattr(x, 'shape'):
    x_shape = x.shape
if hasattr(y, 'shape'):
    y_shape = y.shape

print(x_shape) #Output: (4140, 15) <--- 4140 rows, 15 columns
print(y_shape) #Output: (4140, ) <--- 4140 rows, N/A

knn.fit(x,y) #Downfall of KNN, you have to retrain everytime, does not save model
yhat = knn.predict(x)
print('Prediction value of x: ' + str(yhat[-1]), ', real target: ' + str(y[-1]))
knn.score(x,y)
print(knn.score(x,y)) #Print the score

plt.pie(np.c_[np.sum(np.where(y==1,1,0)),np.sum(np.where(y==-1,1,0))][0],
        labels=['Not fully funded', 'Full amount'],
        colors=['g', 'r'],
        shadow = False,
        autopct = '%.2f' # Floating point, %.2f would be x.xx, %.3f would be x.xxx
        )
plt.gcf().set_size_inches((6,6))
plt.savefig('pie.png',dpi=300,bbox_inches='tight')

# Confusion Matrix
TP = np.sum(np.logical_and(yhat == -1, y == -1))    # True Positive
TN = np.sum(np.logical_and(yhat == 1, y == 1))      # True Negative
FP = np.sum(np.logical_and(yhat == -1, y == 1))     # False Positive
FN = np.sum(np.logical_and(yhat == 1, y == -1))     # False Negative
print('TP: ' + str(TP) + ", FP: " + str(FP))
print('TN: ' + str(TN) + ", FN: " + str(FN))

confusionMatrix = metrics.confusion_matrix(yhat, y)

print("Confusion Matrix: ")
print(confusionMatrix)