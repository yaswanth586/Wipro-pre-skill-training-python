# Perform Data Cleaning and Transformation Tasks
# Load a dataset with missing values.
# Identify the columns with missing values.
# Fill the missing values with the mean of the column.
# Load a dataset where numerical columns are mistakenly read as strings.
# Convert the columns to appropriate data types.
# Load a dataset with duplicate rows.
# Remove the duplicate rows and display the cleaned DataFrame.

import pandas as pd
import numpy as np


def create_data():
    data_with_missing = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'age': [28, np.nan, 22, 35, np.nan],
        'score': [85, 90, np.nan, 88, 92]
    })

    data_with_strings = pd.DataFrame({
        'id': ['1', '2', '3', '4', '5'],
        'age': ['28', '34', '22', '35', '29'],
        'score': ['85', '90', '88', '92', '95']
    })

    data_with_duplicates = pd.DataFrame({
        'id': [1, 2, 2, 4, 5, 5],
        'age': [28, 34, 34, 35, 29, 29],
        'score': [85, 90, 90, 88, 92, 92]
    })

    return data_with_missing, data_with_strings, data_with_duplicates


def clean_missing_values(data_with_missing):
    missing_columns = data_with_missing.columns[data_with_missing.isnull().any()]
    data_filled = data_with_missing.fillna(data_with_missing.mean())

    return data_filled


def value_data_type(data_with_strings):
    data_converted = data_with_strings.astype({'id': int, 'age': int, 'score': float})

    return data_converted


def remove_duplicates(data_with_duplicates):
    data_cleaned = data_with_duplicates.drop_duplicates()

    return data_cleaned


if __name__ == "__main__":
    data_with_missing, data_with_strings, data_with_duplicates = create_data()

    data_filled = clean_missing_values((data_with_missing))
    print("DataFrame after filling missing values with the mean:\n", data_filled)

    data_converted = value_data_type(data_with_strings)
    print("DataFrame after converting data types:\n", data_converted.dtypes)

    data_cleaned = remove_duplicates(data_with_duplicates)
    print("DataFrame after removing duplicate rows:\n", data_cleaned)
