import pandas as pd #pip install pandas

file = open('../../DataFiles/adult.data', 'r')

def chr_int(a):
    if a.isdigit():
        return int(a)
    else:
        return 0    

data = []
for line in file:
    data1 = line.split(',')
    # print(len(data1))
    if len(data1) == 15:
        data.append([chr_int(data1[0]), data1[1], chr_int(data1[2]), 
                     data1[3], chr_int(data1[4]), data1[5], 
                     data1[6], data1[7], data1[8], data1[9], 
                     chr_int(data1[10]), chr_int(data1[11]), chr_int(data1[12]), 
                     data1[13], data1[14].replace('\n', '')])
                     
print(data[1:4])
df = pd.DataFrame(data)
print(df.shape)
counts = df.groupby(df[13]).size()
print(counts)
ml = df.groupby(df[9]).size()
print(ml)
df1 = df[df[14] == ' <=50K']
print("Percent of Number of people with a salary <=50K: ",int(len(df1))/float(len(df)))


df1 = df[df[14] == ' >50K']
print("Percent of Number of people with a salary >50K: ",int(len(df1))/float(len(df)))
df1 = df[df[14] == ' >50K']
print("The Number of people with income >50K: ",(int(len(df1))/int(len(df))) * 100)
print("The average age mean is :", df1[0].mean())
print("The average age var is:", df[0].var())
print("The std age is:", df1[0].std())
print("The median age is:", df1[0].median())
print("The min age is:", df1[0].min())
print("The max age is:", df1[0].max())
age_col = df1[0]
chart = age_col.hist(histtype='stepfilled', bins = 20)
print(chart)
