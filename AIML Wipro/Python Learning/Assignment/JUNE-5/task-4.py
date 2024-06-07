# One-Way ANOVA
# Objective: Perform a one-way ANOVA to compare means across multiple groups.
# Generate three sample datasets each with 20 random values from normal distributions with means of 50, 55, and 60, and a standard deviation of 10.
# Perform a one-way ANOVA to check if there are any significant differences in means across the three groups.

import numpy as np
from scipy.stats import f_oneway

np.random.seed(42)

sample1_data = np.random.normal(loc=50, scale=10, size=20)
sample2_data = np.random.normal(loc=55, scale=10, size=20)
sample3_data = np.random.normal(loc=60, scale=10, size=20)

f_statistic, p_value = f_oneway(sample1_data, sample2_data, sample3_data)

print("F-statistic:", f_statistic)
print("P-value:", p_value)

if p_value < 0.05:
    print("There are significant differences in means across the three groups.")
else:
    print("There are no significant differences in means across the three groups.")
