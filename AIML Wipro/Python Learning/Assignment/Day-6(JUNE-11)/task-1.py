# Handling Missing Data and Scaling Features
# Load the Titanic dataset from a CSV file.
# Identify and handle missing values in the Age, Embarked, and Cabin columns using different imputation methods.
# Standardize the numerical features (Age, Fare) using StandardScaler.
# Normalize the numerical features using MinMaxScaler.
# Compare the distributions of the scaled features using histograms.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load the Titanic dataset
file_path = r'./Titanic-Dataset.csv'
titanic_df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(titanic_df.head())

# Check for missing values
print(titanic_df.isnull().sum())

# Impute missing values in Age with the mean
titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].mean())

# Impute missing values in Embarked with the mode
titanic_df['Embarked'] = titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0])

# Impute missing values in Cabin with a placeholder
titanic_df['Cabin'] = titanic_df['Cabin'].fillna('Unknown')

# Standardize the numerical features
scaler = StandardScaler()
titanic_df[['Age_standardized', 'Fare_standardized']] = scaler.fit_transform(titanic_df[['Age', 'Fare']])

# Normalize the numerical features
minmax_scaler = MinMaxScaler()
titanic_df[['Age_normalized', 'Fare_normalized']] = minmax_scaler.fit_transform(titanic_df[['Age', 'Fare']])

# Plot histograms to compare the distributions
fig, axs = plt.subplots(3, 2, figsize=(12, 18))

# Original distributions
axs[0, 0].hist(titanic_df['Age'], bins=40, color='blue', alpha=0.7)
axs[0, 0].set_title('Original Age Distribution')
axs[0, 1].hist(titanic_df['Fare'], bins=30, color='blue', alpha=0.7)
axs[0, 1].set_title('Original Fare Distribution')

# Standardized distributions
axs[1, 0].hist(titanic_df['Age_standardized'], bins=30, color='green', alpha=0.7)
axs[1, 0].set_title('Standardized Age Distribution')
axs[1, 1].hist(titanic_df['Fare_standardized'], bins=30, color='green', alpha=0.7)
axs[1, 1].set_title('Standardized Fare Distribution')

# Normalized distributions
axs[2, 0].hist(titanic_df['Age_normalized'], bins=30, color='red', alpha=0.7)
axs[2, 0].set_title('Normalized Age Distribution')
axs[2, 1].hist(titanic_df['Fare_normalized'], bins=30, color='red', alpha=0.7)
axs[2, 1].set_title('Normalized Fare Distribution')

plt.tight_layout()
plt.show()
