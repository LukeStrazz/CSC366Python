from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
#Data taken from http://lib.stat.cmu.edu/datasets/boston
raw_file = pd.read_csv("boston_data", sep="\s+", skiprows=22)
data = np.hstack(raw_file[::2,:],raw_file.values[1::2,:2])
target = raw_file.values[1::2,2]
print(data)
print(target)
X_boston, = data
Y_boston, = target

print("Feature names: {}".format(raw_file.describe))
plt.hist(Y_boston)
plt.xlabel("Price")
plt.ylabel("Count")

X = raw_file['LSTAT'].values.reshape(-1,1)
Y = raw_file['MEDV'].values
print(X)
print(Y)
plt.plot(X,Y,"o",alpha=0.5)
plt.show()

sns.lmplot(x="LSTAT", y="MEDV", data=raw_file,height=5.2, aspect=2)
plt.show()

sns.lmplot(x="RM", y="MEDV", data=raw_file,height=5.2, aspect=2)
plt.show()

sns.lmplot(x="CRIM", y="MEDV", data=raw_file,height=5.2, aspect=2)
plt.show()