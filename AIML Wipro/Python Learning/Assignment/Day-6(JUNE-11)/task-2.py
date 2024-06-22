# Encoding Categorical Variables and Feature Engineering
# Load the Iris dataset from a CSV file.
# Perform one-hot encoding on the Species column.
# Perform label encoding on the Species column and compare the results.
# Create a new feature PetalArea by multiplying PetalLength and PetalWidth.
# Create a new feature SepalArea by multiplying SepalLength and SepalWidth.

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the Iris dataset
file_path = r'./Iris.csv'  # Replace with your actual file path
iris_df = pd.read_csv(file_path)

# Perform one-hot encoding on the Species column
one_hot_encoded_df = pd.get_dummies(iris_df, columns=['Species'])

# Perform label encoding on the Species column
label_encoded_df = iris_df.copy()
le = LabelEncoder()
label_encoded_df['Species'] = le.fit_transform(label_encoded_df['Species'])

# Create new features PetalArea and SepalArea
iris_df['PetalArea'] = iris_df['PetalLength'] * iris_df['PetalWidth']
iris_df['SepalArea'] = iris_df['SepalLength'] * iris_df['SepalWidth']

# Display the results
print("Original DataFrame with new features PetalArea and SepalArea:\n")
print(iris_df.head())

print("One-hot Encoded DataFrame:\n")
print(one_hot_encoded_df.head())

print("Label Encoded DataFrame:\n")
print(label_encoded_df.head())
