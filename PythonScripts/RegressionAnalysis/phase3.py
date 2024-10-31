import pandas as pd
ice = pd.read_csv('../../DataFiles/SeaIce.txt', delim_whitespace=True)
print('Shape:', ice.shape)
print(ice.head())
print(ice.mode(numeric_only=True).mean())