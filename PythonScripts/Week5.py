import pickle 
from sklearn import neighbors
from sklearn import datasets
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