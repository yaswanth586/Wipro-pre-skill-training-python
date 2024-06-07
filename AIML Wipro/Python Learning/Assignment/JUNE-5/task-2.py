# Two-Sample t-Test
# Perform a two-sample t-test to compare the means of two independent samples.
# Generate two sample datasets each with 25 random values from normal distributions with means of 55 and 60, and a standard deviation of 8.
# Perform an independent two-sample t-test to check if the means of the two samples are significantly different.

import numpy as np
from scipy import stats

np.random.seed(42)

sample1_data = np.random.normal(loc=55, scale=8, size=25)
sample2_data = np.random.normal(loc=60, scale=8, size=25)

t_statistic, p_value = stats.ttest_ind(sample1_data, sample2_data)

print("T-statistic:", t_statistic)
print("P-value:", p_value)

if p_value < 0.05:
    print("The means of the two samples are significantly different.")
else:
    print("The means of the two samples are not significantly different.")

