import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

file = open('../../DataFiles/ACCIDENTS_GU_BCN_2010.csv', 'r')

data = np.random.normal(100, 15, size=100)
mean_estimate = np.mean(data)
print(f'Mean Estimate: {mean_estimate}')

std_error = np.std(data) / np.sqrt(len(data))
print(f'Standard Error: {std_error}')

confidence_interval = stats.norm.interval(0.95, loc=mean_estimate, scale=std_error)
print(f'95% Confidence Interval: {confidence_interval}')

population_mean = 105
t_stat, p_value = stats.ttest_1samp(data, population_mean)
print(f'T-statistic: {t_stat}, P-value: {p_value}')

data = pd.read_csv("ACCIDENTS_GU_BCN_2013.csv")
data['Date '] = data[u'Dia de mes']. apply(lambda x: str(x))
+ '-' +
data[u'Mes de any']. apply(lambda x: str(x))
data['Date '] = pd.to_datetime(data['Date '])
accidents = data.groupby (['Date ']).size ()

print accidents.mean ()

# population
df = accidents.to_frame ()
N_test = 10000
elements = 200
# mean array of samples
means = [0] * N_test
# sample generation
for i in range(N_test):
rows = np.random.choice(df.index.values , elements)
sampled_df = df.ix[rows]
means[i] = sampled_df.mean ()

rows = np.random.choice(df.index.values , 200)
sampled_df = df.ix[rows]
est_sigma_mean = sampled_df.std()/math.sqrt (200)
print 'Direct estimation of SE from one sample of
200 elements:', est_sigma_mean [0]
print 'Estimation of the SE by simulating 10000 samples of
200 elements:', np.array(means).std()

def meanBootstrap(X, numberb):
    x = [0]* numberb
    for i in range(numberb):
        sample = [X[j]
            for j
                in np.random.randint(len(X), size=len(X))
            ]
        x[i] = np.mean(sample)
    return x

m = meanBootstrap(accidents , 10000)
print "Mean estimate:", np.mean(m)

m = accidents.mean ()
se = accidents.std()/math.sqrt(len(accidents))
ci = [m - se*1.96 , m + se *1.96]
print "Confidence interval:", ci

data = pd.read_csv("ACCIDENTS_GU_BCN_2010.csv", encoding='latin -1')
# Create a new column which is the date
data['Date'] = data['Dia de mes']. apply(lambda x: str(x)) + '-' + data['Mes de any']. apply(lambda x: str(x))
data2 = data['Date']
counts2010 = data['Date']. value_counts()
print '2010: Mean', counts2010.mean()
data = pd.read_csv("ACCIDENTS_GU_BCN_2013.csv", encoding='latin -1')
# Create a new column which is the date
data['Date'] = data['Dia de mes']. apply(lambda x: str(x)) + '-' + data['Mes de any']. apply(lambda x: str(x))
data2 = data['Date']
counts2013 = data['Date']. value_counts()
print '2013: Mean', counts2013.mean()

n = len(counts2013)
mean = counts2013.mean ()
s = counts2013.std()
ci = [mean - s*1.96/ np.sqrt(n), mean + s*1.96/ np.sqrt(n)]
print '2010 accident rate estimate:', counts2010.mean ()
print '2013 accident rate estimate:', counts2013.mean ()
print 'CI for 2013: ',ci

plt.hist(data, bins=20, alpha=0.7)
plt.axvline(x=population_mean, color='red', linestyle='--')
plt.title('Sample Data Distribution')
plt.show()

m= len(counts2010)
n= len(counts2013)
p = (counts2013.mean () - counts2010.mean ())
print 'm:', m, 'n:', n
print 'mean difference: ', p

# pooling distributions
x = counts2010
y = counts2013
pool = np.concatenate ([x, y])
np.random.shuffle(pool)
#sample generation
import random
N = 10000 # number of samples
diff = range(N)
for i in range(N):
    p1 = [random.choice(pool) for _ in xrange(n)]
    p2 = [random.choice(pool) for _ in xrange(n)]
    diff[i] = (np.mean(p1) - np.mean(p2))

# counting differences larger than the observed one
diff2 = np.array(diff)
w1 = np.where(diff2 > p)[0]

print 'p-value (Simulation)=', len(w1)/float(N), '(', len(w1)/float(N)*100
,'%)', 'Difference =', p

if (len(w1)/float(N)) < 0.05:
    print 'The effect is likely '
else:
    print 'The effect is not likely '
