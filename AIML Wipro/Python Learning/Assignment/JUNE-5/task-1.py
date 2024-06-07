# One-Sample t-Test
# Perform a one-sample t-test to determine if the sample mean is significantly different from a known population mean.
# Generate a sample dataset of 30 random values from a normal distribution with a mean of 60 and a standard deviation of 10.
# Perform a one-sample t-test to check if the sample mean is significantly different from 50.

import numpy as np
from scipy import stats

np.random.seed(42)
sample_data = np.random.normal(loc=60, scale=10, size=30)

t_statistic, p_value = stats.ttest_1samp(sample_data, 50)

# Output results
print("T-statistic:", t_statistic)
print("P-value:", p_value)

if p_value < 0.05:
    print("The sample mean is significantly different from 50.")
else:
    print("The sample mean is not significantly different from 50.")

