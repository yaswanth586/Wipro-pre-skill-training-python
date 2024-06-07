# Chi-Squared Test
# Objective: Perform a Chi-Squared test for independence.
# Create a contingency table with observed frequencies for two categorical variables.
#
# |-----------| Category A | Category B |
# | Group 1 |     10   	   |     20   	|
# | Group 2 |     15       |     25     |
#
# Perform a Chi-Squared test to determine if there is a significant association between the two categorical variables.

import numpy as np
from scipy.stats import chi2_contingency

observed = np.array([[10, 20],
                     [15, 25]])

chi2_stat, p_val, dof, expected = chi2_contingency(observed)

print("Chi-Squared Statistic:", chi2_stat)
print("P-value:", p_val)

if p_val < 0.05:
    print("There is a significant association between the two categorical variables.")
else:
    print("There is no significant association between the two categorical variables.")
