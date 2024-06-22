# Merge and Join Multiple DataFrames
# Load two DataFrames, df1 and df2, with a common column id.
# Perform an inner join on the id column.
# Display the merged DataFrame.
# Load two DataFrames, df1 and df2, with a common column id.
# Perform an outer join on the id column.
# Display the merged DataFrame.
# Load two DataFrames with the same columns.
# Concatenate the DataFrames vertically.
# Display the concatenated DataFrame.

import pandas as pd


def inner_join(df1, df2):
    inner_join_df = pd.merge(df1, df2, on='id', how='inner')

    return inner_join_df


def outer_join(df1, df2):
    outer_join_df = pd.merge(df1, df2, on='id', how='outer')

    return outer_join_df


def concatenate_df_vertically(df3, df4):
    concatenated_df = pd.concat([df3, df4], axis=0)

    return concatenated_df


if __name__ == "__main__":
    df1 = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'name': ['yaswanth', 'shiva', 'satya', 'satish'],
        'age': [25, 30, 35, 40]
    })
    df2 = pd.DataFrame({
        'id': [3, 4, 5, 6],
        'score': [88, 92, 85, 90],
        'city': ['Andhra Pradesh', 'Uttar Pradesh', 'Tamil Nadu', 'Maharashtra']
    })

    inner_join_df = inner_join(df1, df2)
    print("Inner Join DataFrame:\n", inner_join_df)

    outer_join_df = outer_join(df1, df2)
    print("Outer Join DataFrame:\n", outer_join_df)

    df3 = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['yaswanth', 'shiva', 'satya',],
        'age': [25, 30, 35]
    })
    df4 = pd.DataFrame({
        'id': [4, 5, 6],
        'name': ['satish', 'ravi', 'faizan'],
        'age': [40, 45, 50]
    })

    concatenated_df = concatenate_df_vertically(df3, df4)
    print("Concatenated DataFrame:\n", concatenated_df)
