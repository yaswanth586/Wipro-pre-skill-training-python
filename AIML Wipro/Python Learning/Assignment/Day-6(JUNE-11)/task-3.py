# Data Visualization
# Load the Titanic dataset.
# Create box plots to identify outliers in the Age and Fare columns.
# Create histograms and KDE plots to visualize the distribution of Age and Fare.
# Create scatter plots to visualize the relationship between Age and Fare, and Pclass and Survived.
# Use pair plots to visualize the relationships between multiple numerical features.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset from a CSV file
titanic = pd.read_csv(r'./Titanic-Dataset.csv')

# Capitalize the first letter of column names
titanic.columns = [col.capitalize() for col in titanic.columns]

# Display the first few rows of the dataset to verify it loaded correctly
print(titanic.head())

# Step 1: Box Plots to Identify Outliers
plt.figure(figsize=(14, 6))

# Box plot for Age
plt.subplot(1, 2, 1)
sns.boxplot(x=titanic['Age'])
plt.title('Box Plot of Age')

# Box plot for Fare
plt.subplot(1, 2, 2)
sns.boxplot(x=titanic['Fare'])
plt.title('Box Plot of Fare')

plt.show()

# Step 2: Histograms and KDE Plots
plt.figure(figsize=(14, 12))

# Histogram and KDE for Age
plt.subplot(2, 2, 1)
sns.histplot(titanic['Age'].dropna(), kde=True)
plt.title('Histogram and KDE of Age')

# Histogram and KDE for Fare
plt.subplot(2, 2, 2)
sns.histplot(titanic['Fare'], kde=True)
plt.title('Histogram and KDE of Fare')

plt.show()

# Step 3: Scatter Plots
plt.figure(figsize=(14, 6))

# Scatter plot for Age vs Fare
plt.subplot(1, 2, 1)
sns.scatterplot(x=titanic['Age'], y=titanic['Fare'])
plt.title('Scatter Plot of Age vs Fare')

# Scatter plot for Pclass vs Survived
plt.subplot(1, 2, 2)
sns.scatterplot(x=titanic['Pclass'], y=titanic['Survived'])
plt.title('Scatter Plot of Pclass vs Survived')

plt.show()

# Step 4: Pair Plots
sns.pairplot(titanic[['Age', 'Fare', 'Pclass', 'Survived']], hue='Survived')
plt.show()
