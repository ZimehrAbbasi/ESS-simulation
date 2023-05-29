import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read in the data
banking = pd.read_csv('banks.csv')

# Print out the column labels
print(banking.columns)

# keep only column A and L
banking_subset = banking[['A', 'L']] * 1000
A = banking_subset['A']
L = banking_subset['L']

mean_A = np.mean(A)
sd_A = np.std(A)
mean_L = np.mean(L)
sd_L = np.std(L)

logmeanA = np.log(mean_A**2 / np.sqrt(sd_A**2 + mean_A**2))
logmeanL = np.log(mean_L**2 / np.sqrt(sd_L**2 + mean_L**2))
logstdA = np.sqrt(np.log(sd_A**2 / mean_A**2 + 1))
logstdL = np.sqrt(np.log(sd_L**2 / mean_L**2 + 1))

# print matrix of mean and sd
print(np.array([[logmeanA, logstdA], [logmeanL, logstdL]]))

# remove all rows with values smaller than 1
banking_subset = banking_subset[banking_subset > 1]
# remove all rows with missing values
banking_subset = banking_subset.dropna()

# find mean and sd of log A and L base 10
banking_subset['logA'] = np.log(banking_subset['A'])
banking_subset['logL'] = np.log(banking_subset['L'])

# display the density plot of logA
# banking_subset['logA'].plot(kind='density')
# plt.show()

# sample from logA and logL not normal distribution
logA = banking_subset['logA']
logL = banking_subset['logL']

mean_logA = np.mean(logA)
sd_logA = np.std(logA)
mean_logL = np.mean(logL)
sd_logL = np.std(logL)

# sample from logA and logL lognormal distribution
logA_sample = np.random.lognormal(mean_logA, sd_logA, 100)
logL_sample = np.random.lognormal(mean_logL, sd_logL, 1)

print(np.mean(np.log(logA_sample)))