import pandas as pd
import numpy as np

# Read in the data
banking = pd.read_csv('banks.csv')

# Print out the column labels
print(banking.columns)

# keep only column A and L
banking_subset = banking[['A', 'L']]

# find mean and sd of log A and L
banking_subset['logA'] = np.log(banking_subset['A'])
banking_subset['logL'] = np.log(banking_subset['L'])

# remove all rows with values smaller than 1
banking_subset = banking_subset[banking_subset > 1]

# print out the mean and sd of log A and L
print(banking_subset[['logA', 'logL']].agg([np.mean, np.std]))