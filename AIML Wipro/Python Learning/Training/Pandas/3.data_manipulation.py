"""
Data Manipulation with Pandas
"""

# Reading Data from CSV, Excel, JSON, and SQL

import pandas as pd

# Creating a sample DataFrame
data = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8],
    'C': [9, 10, 11, 12]
})

data.to_csv('sample_data.csv', index=False)
data.to_excel('sample_data.xlsx', index=False)
data.to_json('sample_data.json', orient='records')

# Reading data

csv_data = pd.read_csv('sample_data.csv')
print("Data from CSV:\n", csv_data)

excel_data = pd.read_excel('sample_data.xlsx')
print("Data from Excel:\n", excel_data)

json_data = pd.read_json('sample_data.json')
print("Data from JSON:\n", json_data)

import sqlite3

# Create an in-memory SQLite database and insert the sample data
conn = sqlite3.connect(':memory:')
data.to_sql('sample_table', conn, index=False, if_exists='replace')

# Reading data from SQL
sql_data = pd.read_sql('SELECT * FROM sample_table', conn)
print("Data from SQL:\n", sql_data)


# ==========================================================================

# Selecting Data by Labels (.loc)

import pandas as pd

# Sample DataFrame
data = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8],
    'C': [9, 10, 11, 12]
}, index=['row1', 'row2', 'row3', 'row4'])
print(data)
# Selecting rows with label 'row2' and specific columns 'A' and 'C'
selected_data = data.loc['row2', ['A', 'C']]
print(selected_data)

# Selecting the first 2 rows and first 2 columns
selected_data = data.iloc[0:2, 0:2]
print(selected_data)

# Selecting rows where column 'A' is greater than 2
filtered_data = data[data['A'] > 2]
print(filtered_data)


'''
Data Cleaning
Handling Missing Data (dropna, fillna)
'''

# Sample DataFrame with missing values
data = pd.DataFrame({
    'A': [1, 2, None, 4],
    'B': [None, 2, 3, 4],
    'C': [1, 2, 3, None]
})

# Dropping rows with any missing values
cleaned_data_drop = data.dropna()
print('cleaned \n',cleaned_data_drop)

# Filling missing values with 0
cleaned_data_fill = data.fillna(0)
print(cleaned_data_fill)

# Sample DataFrame with duplicates
data = pd.DataFrame({
    'A': [1, 2, 2, 4],
    'B': [5, 6, 6, 8]
})

# Removing duplicate rows
cleaned_data = data.drop_duplicates()
print(cleaned_data)

# Sample DataFrame
data = pd.DataFrame({
    'A': ['1', '2', '3', '4']
})
print('before \n',data)
print(data.dtypes)
# Converting data type of column 'A' to integer
data['A'] = data['A'].astype(int)
print('after \n',data)
print(data.dtypes)

# Sample DataFrame
data = pd.DataFrame({
    'A': ['Hello', 'World', 'Pandas', 'Python']
})

# Converting column 'A' to lowercase
data['A'] = data['A'].str.lower()
print(data)


'''
Data Transformation
Applying Functions to Data (apply, map, applymap)
'''

# Sample DataFrame
data = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8]
})

# Applying a function to column 'A'
data['A'] = data['A'].apply(lambda x: x * 2)
print(data)

# Mapping values in column 'B'
data['B'] = data['B'].map({5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight'})
print(data)

# Applying a function to the entire DataFrame using map
data = data.applymap(str)
print('str tra', data)
print(data.dtypes)

# Sample DataFrame
data = pd.DataFrame({
    'A': [3, 1, 4, 2],
    'B': [5, 6, 7, 8]
})

# Sorting data by column 'A'
sorted_data = data.sort_values(by='A')
print(sorted_data)

data = pd.DataFrame({
    'A': [3, 1, 2, 2],
    'B': [5, 6, 7, 8]
})

# Ranking data in column 'A'
data['rank'] = data['A'].rank()
print(data)

# Sample DataFrame
data = pd.DataFrame({
    'A': [5, 15, 25, 35, 45, 4, 44,23,38,24]
})

# Binning data into discrete intervals
bins = [0, 10, 20, 30, 40, 50]
labels = ['0-10', '10-20', '20-30', '30-40', '40-50']
data['binned'] = pd.cut(data['A'], bins=bins, labels=labels)
print(data)

'''
Merging and Joining DataFrames
'''

import pandas as pd

# Creating sample DataFrames
df1 = pd.DataFrame({
    'A': ['A0', 'A1', 'A2', 'A3'],
    'B': ['B0', 'B1', 'B2', 'B3'],
    'C': ['C0', 'C1', 'C2', 'C3'],
    'D': ['D0', 'D1', 'D2', 'D3']
}, index=[0, 1, 2, 3])

df2 = pd.DataFrame({
    'A': ['A4', 'A5', 'A6', 'A7'],
    'B': ['B4', 'B5', 'B6', 'B7'],
    'C': ['C4', 'C5', 'C6', 'C7'],
    'D': ['D4', 'D5', 'D6', 'D7']
}, index=[4, 5, 6, 7])

# Concatenating DataFrames
result = pd.concat([df1, df2])
print("Concatenated DataFrame:\n", result)

# Merging on keys (merge)

left = pd.DataFrame({
    'key': ['K0', 'K1', 'K2', 'K3'],
    'A': ['A0', 'A1', 'A2', 'A3'],
    'B': ['B0', 'B1', 'B2', 'B3']
})

right = pd.DataFrame({
    'key': ['K0', 'K1', 'K4', 'K5'],
    'C': ['C0', 'C1', 'C2', 'C3'],
    'D': ['D0', 'D1', 'D2', 'D3']
})

# Merging DataFrames
result = pd.merge(left, right, on='key')
print("Merged DataFrame:\n", result)

# Joining DataFrames (join)
# Creating sample DataFrames with different indexes
dfj1 = pd.DataFrame({
    'A': ['A0', 'A1', 'A2'],
    'B': ['B0', 'B1', 'B2']
}, index=['K0', 'K1', 'K2'])

dfj2 = pd.DataFrame({
    'C': ['C0', 'C2', 'C3'],
    'D': ['D0', 'D2', 'D3']
}, index=['K0', 'K2', 'K3'])

# Joining DataFrames
result = dfj1.join(dfj2, how='outer')
print("Joined DataFrame:\n", result)

# Grouping and Aggregation
# Grouping data (groupby)

import pandas as pd
import numpy as np

# Creating a sample DataFrame
df = pd.DataFrame({
    'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
    'B': ['one', 'one', 'two', 'two', 'one', 'one', 'two', 'two'],
    'C': [1, 2, 3, 4, 5, 6, 7, 8],
    'D': np.random.randn(8)
})

# Grouping by column 'A'
grouped = df.groupby('A')
# Aggregating numeric data with mean
mean_result = grouped[['C', 'D']].mean()
print("Grouped DataFrame mean:\n", mean_result)



#Time Series Analysis
#Working with date and time data

# Creating a sample DataFrame with date range
dates = pd.date_range('20230101', periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
print("DataFrame with dates:\n", df)

#Resampling and frequency conversion
# Resampling to a different frequency
resampled = df.resample('M').mean()
print("Resampled DataFrame (monthly mean):\n", resampled)

#Rolling and expanding windows
# Applying rolling window
rolling = df.rolling(window=2).mean()
print("Rolling window (mean):\n", rolling)

# Applying expanding window
expanding = df.expanding(min_periods=1).mean()
print("Expanding window (mean):\n", expanding)


# Visualization with Pandas
# Basic plotting with Pandas
import matplotlib.pyplot as plt

# Creating a sample DataFrame
df = pd.DataFrame(np.random.randn(10, 4), columns=['A', 'B', 'C', 'D'])
# Plotting
df.plot()
plt.title('Basic Plot')
plt.show()
# Customizing plot with labels and title
df.plot()
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.title('Customized Plot')
plt.show()
# Integration with Matplotlib and Seaborn
import seaborn as sns

# Using Seaborn for a more advanced plot
sns.lineplot(data=df)
plt.title('Seaborn Line Plot')
plt.show()


#Missing values
import pandas as pd
import numpy as np

# Creating a sample DataFrame with missing values
data = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, np.nan, 5],
    'C': [1, 2, 3, 4, 5],
    'D': [np.nan, np.nan, np.nan, 4, 5]
}
df = pd.DataFrame(data)
print('Data \n', df)
# Identifying missing data
print("Count of Missing data:\n", df.isnull().sum())

#Techniques to Handle Missing Data

'''
data = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, np.nan, 5],
}
df = pd.DataFrame(data)
'''
# Dropping rows with any missing data
df_dropped_rows = df.dropna(axis=0)
print("DataFrame after dropping rows with any missing data:\n", df_dropped_rows)

# Dropping columns with any missing data
df_dropped_cols = df.dropna(axis=1)
print("DataFrame after dropping columns with any missing data:\n", df_dropped_cols)

# Filling missing data with a specific value
df_filled = df.fillna(0)
print("DataFrame after filling missing data with 0:\n", df_filled)

'''
Imputation Methods
'''

# Imputing missing data with the mean of each column
df_mean_imputed = df.fillna(df.mean())
print("DataFrame after mean imputation:\n", df_mean_imputed)

# Imputing missing data with the median of each column
df_median_imputed = df.fillna(df.median())
print("DataFrame after median imputation:\n", df_median_imputed)

# Imputing missing data with the mode of each column
df_mode_imputed = df.fillna(df.mode().iloc[0])
print("DataFrame after mode imputation:\n", df_mode_imputed)

# Interpolating missing data
df_interpolated = df.interpolate()
print("DataFrame after interpolation:\n", df_interpolated)
