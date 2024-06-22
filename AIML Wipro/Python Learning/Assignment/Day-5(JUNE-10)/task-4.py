# Group and Aggregate Data for Analysis
# Load a dataset with columns category and value.
# Group the data by category and compute the sum of value for each category.
# Display the aggregated DataFrame.
# Load a dataset with columns category and value.
# Group the data by category and compute the mean and standard deviation of value for each category.
# Display the aggregated DataFrame.
# Load a dataset with columns category, subcategory, and value.
# Create a pivot table that shows the sum of value for each combination of category and subcategory.
# Display the pivot table.

import pandas as pd


def group_by_sum(data):
    grouped_sum = data.groupby('category')['value'].sum().reset_index()

    return grouped_sum


def group_by_mean_std(data):
    grouped_stats = data.groupby('category')['value'].agg(['mean', 'std']).reset_index()

    return grouped_stats


def pivot_table(data):
    pivot_table = pd.pivot_table(data, values='value', index='category', columns='subcategory',
                                 aggfunc='sum', fill_value=0)

    return pivot_table


if __name__ == "__main__":
    data = pd.DataFrame({
        'category': ['A', 'B', 'A', 'B', 'C', 'C', 'A'],
        'value': [10, 20, 30, 40, 50, 60, 70]
    })

    grouped_sum = group_by_sum(data)
    print("Aggregated DataFrame with Sum:\n", grouped_sum)

    grouped_stats = group_by_mean_std(data)
    print("Aggregated DataFrame with Mean and Standard Deviation:\n", grouped_stats)

    data_with_subcategory = pd.DataFrame({
        'category': ['A', 'A', 'A', 'B', 'B', 'C', 'C'],
        'subcategory': ['X', 'Y', 'X', 'Y', 'X', 'Y', 'X'],
        'value': [10, 20, 30, 40, 50, 60, 70]
    })

    pivot_table = pivot_table(data_with_subcategory)
    print("Pivot Table:\n", pivot_table)
