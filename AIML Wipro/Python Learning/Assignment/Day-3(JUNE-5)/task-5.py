# Post-hoc Test using Tukey's HSD
# Objective: Perform a post-hoc test using Tukey's HSD to identify which groups are significantly different.
# Use the same datasets generated in the one-way ANOVA exercise.
# Perform Tukey's HSD test to find out which pairs of group means are significantly different.

import numpy as np
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

sample1_data = np.random.normal(loc=50, scale=10, size=20)
sample2_data = np.random.normal(loc=55, scale=10, size=20)
sample3_data = np.random.normal(loc=60, scale=10, size=20)

data = pd.DataFrame({'value': np.concatenate([sample1_data, sample2_data, sample3_data]), 'group': ['Group 1'] * 20 + ['Group 2'] * 20 + ['Group 3'] * 20})
tukey_results = pairwise_tukeyhsd(data['value'], data['group'])

print(tukey_results)
