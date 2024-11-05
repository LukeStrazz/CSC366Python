import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns

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
plt.savefig("phase-3-figure1.png", dpi=300, bbox_inches='tight')

grouped = ice2.groupby('mo')
month_means = grouped.extent.mean()
month_variance = grouped.extent.var()
print('Mean:', month_means)
print('Variance: ', month_variance)

for i in range(12):
    mask = ice2['mo'] == i + 1
    ice.loc[mask, 'extent'] = 100*(ice2.loc[mask, 'extent'] - month_means[i+1])/month_means.mean()
    
sns.lmplot(x="mo", y="extent", data=ice2, aspect=2)
plt.savefig('phase3-figure-2.png')