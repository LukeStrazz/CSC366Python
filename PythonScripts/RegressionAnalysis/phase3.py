import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import scipy.stats
from sklearn.linear_model import LinearRegression
from sklearn import metrics

ice = pd.read_csv('../../DataFiles/SeaIce.txt', delim_whitespace=True)
print('Shape:', ice.shape)
print(ice.head())
print(ice.mode(numeric_only=True).mean())

x = ice.year
y = ice.extent

plt.scatter(x,y,color='red')
plt.xlabel('Year')
plt.ylabel('Extent')
#plt.show()
print('Different values in data fields', np.unique(ice.data_type.values))
print(ice[(ice.data_type != 'Goddard') & (ice.data_type != 'NRTSI-G')])

ice2 = ice[ice.data_type != '-9999']

x = ice2.year
y = ice2.extent
print(x, y)
plt.scatter(x,y,color='green')
plt.xlabel('Year')
plt.ylabel('Extent')
plt.show()

sns.lmplot(x="mo", y="extent", data=ice2, aspect=2)
plt.savefig("phase-3-figure-1.png", dpi=300, bbox_inches='tight')

grouped = ice2.groupby('mo')
month_means = grouped.extent.mean()
month_variance = grouped.extent.var()
print('Mean:', month_means)
print('Variance: ', month_variance)

for i in range(12):
    mask = ice2['mo'] == i + 1
    ice2.loc[mask, 'extent'] = 100*(ice2.loc[mask, 'extent'] - month_means[i+1])/month_means.mean()
    
sns.lmplot(x="mo", y="extent", data=ice2, aspect=2)
plt.savefig('phase-3-figure-2.png')

print('Mean: ', ice2.extent.mean())
print('Variance: ', ice2.extent.var())
sns.lmplot(x="year", y="extent", data=ice2, aspect=2)
plt.savefig("phase-3-figure-3.png")
jan = ice2[ice2.mo == 1]
sns.lmplot(x="year", y="extent", data=jan, height=6, aspect=2)
plt.savefig("phase-3-figure-4.png")
aug = ice2[ice2.mo == 8]
sns.lmplot(x="year", y="extent", data=aug, height=6, aspect=2)
plt.savefig("phase-3-figure-5.png")

print(scipy.stats.pearsonr(ice2.year.values, ice2.extent.values))

est = LinearRegression(fit_intercept= True)
x = ice2[['year']]
y = ice2[['extent']]
est.fit(x, y)

print("Coefficient: ", est.coef_)
print("Intercept: ", est.intercept_)

y_hat = est.predict(x)
plt.plot(x, y, 'o', alpha=0.5)
plt.plot(x, y_hat, 'r', alpha=0.5)
plt.xlabel("Year")
plt.ylabel("Extent")
print("Mean Square Error: ", metrics.mean_squared_error(y_hat, y))
print("R^2: ", metrics.r2_score(y, y_hat)) 
print("Variacnce: ", y.var()) 

x_jan = jan[['year']]
y_jan = jan[['extent']]
jan_model = LinearRegression()
jan_model.fit(x_jan, y_jan)
y_jan_hat = jan_model.predict(x_jan)

plt.figure()
plt.plot(x_jan, y_jan, '-o', alpha=0.5)
plt.plot(x_jan, y_jan_hat, 'g', alpha=0.5)
plt.xlabel('year')
plt.ylabel('extent')
plt.show()

