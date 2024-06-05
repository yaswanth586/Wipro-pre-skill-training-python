# Descriptive Statistics with NumPy and SciPy
#
# Create a dataset with 20 random values between 1 and 100.
# Compute the following statistics for the dataset:
# Mean
# Median
# Standard deviation
# Variance
# Skewness
# Kurtosis

import numpy as np
from scipy import stats

# generate a dataset with 20 random values between 1 and 100
dataset = np.random.randint(1, 101, size=20)

# mean
mean = np.mean(dataset)

# median
median = np.median(dataset)

# standard deviation
std_dev = np.std(dataset)

# variance
variance = np.var(dataset)

# skewness
skewness = stats.skew(dataset)

# kurtosis
kurtosis = stats.kurtosis(dataset)

print("Dataset:", dataset)
print("Mean:", mean)
print("Median:", median)
print("Standard Deviation:", std_dev)
print("Variance:", variance)
print("Skewness:", skewness)
print("Kurtosis:", kurtosis)
