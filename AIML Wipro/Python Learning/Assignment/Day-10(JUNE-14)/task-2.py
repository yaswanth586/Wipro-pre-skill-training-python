# Implement the k-Nearest Neighbors (k-NN) algorithm to
# classify the Wine Quality dataset into good and bad quality wines using different values of k.
# Evaluate the model's performance using accuracy.
# Steps:
# Load the Wine Quality dataset from the UCI Machine Learning Repository.
# Preprocess the data: normalize the features and binarize
# the target variable into good (quality >= 7) and bad (quality < 7).
# Split the data into training and testing sets.
# Implement the k-NN algorithm
# Experiment with different values of k (e.g., k=3, 5, 7) and evaluate the model's performance.
# Report the accuracy for each value of k

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(url, delimiter=';')

# Preprocess the data: binarize the target variable
data['quality'] = data['quality'].apply(lambda x: 1 if x >= 7 else 0)

X = data.drop('quality', axis=1).values
y = data['quality'].values

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Function to evaluate k-NN with different values of k
def evaluate_knn(k_values):
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'k = {k}, Accuracy: {accuracy:.4f}')


# Experiment with different values of k
k_values = [3, 5, 7, 9, 11]
evaluate_knn(k_values)
