# Implement k-Means Clustering on a Dataset using PyTorch
# Dataset: Iris Dataset
# Objective: Implement k-Means clustering on the Iris dataset to group the data into clusters.
# Visualize the results and evaluate the clustering performance using a suitable metric such as silhouette score.
# Steps:
# Load the Iris dataset:
# Use pandas to load the Iris dataset from the UCI Machine Learning Repository.
# Preprocess the data:
# Normalize the features using StandardScaler.
# Implement k-Means Clustering using PyTorch:
# Initialize cluster centroids randomly.
# Assign each data point to the nearest centroid.
# Update centroids by computing the mean of the assigned points.
# Repeat the process until convergence.
# Visualize the Clusters:
# Use matplotlib to plot the clusters.
# If the data has more than 2 features, use PCA to reduce dimensionality before plotting.
# Evaluate the Clustering Performance:
# Use silhouette score from sklearn.metrics.


import pandas as pd
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to PyTorch tensor
X = torch.tensor(X, dtype=torch.float32)


# Define k-means clustering
class KMeansClustering:
    def __init__(self, n_clusters, max_iters=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, X):
        # Randomly initialize cluster centers
        self.centroids = X[torch.randint(len(X), (self.n_clusters,))]
        for _ in range(self.max_iters):
            # Compute distances and assign clusters
            distances = torch.cdist(X, self.centroids)
            self.labels = torch.argmin(distances, dim=1)

            # Update centroids
            new_centroids = torch.stack([X[self.labels == i].mean(dim=0) for i in range(self.n_clusters)])
            # Check for convergence
            if torch.norm(self.centroids - new_centroids) < self.tol:
                break
            self.centroids = new_centroids

    def predict(self, X):
        distances = torch.cdist(X, self.centroids)
        return torch.argmin(distances, dim=1)


# Apply k-means clustering
kmeans = KMeansClustering(n_clusters=3)
kmeans.fit(X)
labels = kmeans.predict(X)

# Evaluate the clustering performance
silhouette_avg = silhouette_score(X, labels)
print(f'Silhouette Score: {silhouette_avg:.4f}')

# Visualize the clusters using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.numpy())

plt.figure(figsize=(10, 6))
for i in range(3):
    plt.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], label=f'Cluster {i}')
plt.scatter(pca.transform(kmeans.centroids.numpy())[:, 0], pca.transform(kmeans.centroids.numpy())[:, 1], color='black', marker='x', s=100, label='Centroids')
plt.legend()
plt.title('k-Means Clustering on Iris Dataset')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.show()
