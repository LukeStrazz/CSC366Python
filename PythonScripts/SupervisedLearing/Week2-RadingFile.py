import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

edu = pd.read_csv('../../DataFiles/educ_figdp_1_Data.csv',
na_values = ':',
usecols = ["TIME","GEO","Value"])
print(edu)
print(edu.head())
print(edu.tail())
print(edu.describe())

edu.head()
edu.tail()

edu.describe()

edu['Value']

edu[10:14]

#edu.ix[90:94 , ['TIME ','GEO']]

print(edu.iloc[90:94, [edu.columns.get_loc('TIME'), edu.columns.get_loc('GEO')]])

edu[edu['Value'] > 6.5].tail()

edu[edu["Value"].isnull()].head ()

edu.max(axis = 0)

test = edu.max(axis = 0)

print(test)

test2 = edu.max(axis = 0)

print(test2)

print("Pandas max function", edu['Value'].max())
print("Python max function", max(edu['Value']))

s = edu['Value']/100

print(s)

sqr = edu['Value'].apply(np.sqrt).head()

print(sqr)

exp = edu['Value'].apply(lambda d: d*d*d)

print(exp)

edu['ValueNorm'] = edu['Value'] / edu['Value'].max()

print(edu['ValueNorm'])

edu['ValueNorm'] = edu['Value'] / edu['Value'].max()

print(edu)

edu.drop('ValueNorm', axis = 1, inplace = True)

print(edu)

exit()