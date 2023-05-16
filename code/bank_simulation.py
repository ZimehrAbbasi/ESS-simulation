import numpy as np

# Set the random seed for reproducibility (optional)
np.random.seed(42)


# Function to generate values based on lognormal distribution
def generate_values(mean, std, size):
    return np.random.lognormal(mean=mean, sigma=std, size=size)


# Generate the initial array with random values (mean 1000 and std 500, mean 100 and std 50)
bank_array = np.column_stack(
    (generate_values(7.00306, 0.69115, 100), generate_values(4.60517, 0.92103, 100))
)

# Check the condition for each row and regenerate values if necessary
for i in range(bank_array.shape[0]):
    while bank_array[i, 0] < bank_array[i, 1]:
        bank_array[i, :] = np.column_stack(
            (generate_values(7.00306, 0.69115, 1), generate_values(4.60517, 0.92103, 1))
        )

# Print the resulting array with assets and equity (in millions of dollars)
print(bank_array)

# Manually created external assets with mean and standard deviation
asset_array = np.array([[0.02, 0.01], [0.04, 0.04], [0.06, 0.08], [0.08, 0.12], [0.1, 0.16]])

print(asset_array)
