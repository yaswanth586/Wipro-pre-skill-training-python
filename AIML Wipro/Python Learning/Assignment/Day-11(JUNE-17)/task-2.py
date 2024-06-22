# Apply Hierarchical Clustering and Create a Dendrogram
# Dataset: Wine Quality Dataset
# Objective: Apply hierarchical clustering on the Wine Quality dataset to group the data into clusters.
# Create a dendrogram to visualize the hierarchical relationships.
# Steps:
# Load the Wine Quality dataset:
# Use pandas to load the Wine Quality dataset from the UCI Machine Learning Repository.
# Preprocess the data:
# Normalize the features using StandardScaler.
# Apply Hierarchical Clustering:
# Use scipy.cluster.hierarchy to perform hierarchical clustering.
# Use linkage method to compute the hierarchical clustering.
# Create and Visualize the Dendrogram:
# Use dendrogram function to create the dendrogram.
# Evaluate the Clustering Performance:
# Use silhouette score from sklearn.metrics.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Load the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(url, delimiter=';')

# Preprocess the data: normalize the features
X = data.drop('quality', axis=1).values

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Apply hierarchical clustering
Z = linkage(X, method='ward')

# Create and visualize the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title('Dendrogram for Wine Quality Data')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# Cut the dendrogram to form clusters
max_d = 7.0  # Adjust this value to change the number of clusters
clusters = fcluster(Z, max_d, criterion='distance')

# Evaluate the clustering performance
silhouette_avg = silhouette_score(X, clusters)
print(f'Silhouette Score: {silhouette_avg:.4f}')
