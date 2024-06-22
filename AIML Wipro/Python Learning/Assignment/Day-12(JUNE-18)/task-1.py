# Implement PCA on a Dataset using PyTorch and Visualize the Results
# Dataset: Wine Quality Dataset
# Objective: Implement PCA on the Wine Quality dataset to reduce its dimensionality and visualize the results in 2D.
# Steps:
# Load the Wine Quality dataset:
# Use pandas to load the Wine Quality dataset from the UCI Machine Learning Repository.
# Preprocess the data:
# Normalize the features using StandardScaler.
# Implement PCA using PyTorch:
# Compute the covariance matrix.
# Compute eigenvalues and eigenvectors of the covariance matrix.
# Project the data onto the first two principal components.
# Visualize the Results:
# Use matplotlib to plot the 2D projection of the data.

import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
wine_data = pd.read_csv(url, sep=';')

# Separate features and labels
X = wine_data.drop(columns=['quality'])  # features
y = wine_data['quality']  # labels (not used in PCA)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert numpy array to PyTorch tensor
X_tensor = torch.tensor(X_scaled, dtype=torch.float64)

# Compute the covariance matrix
covariance_matrix = torch.matmul(X_tensor.T, X_tensor) / X_tensor.size(0)

# Compute eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
eigenvalues = eigenvalues.real  # extract only the real part of eigenvalues
eigenvectors = eigenvectors.T  # transpose eigenvectors for easier projection

# Project the data onto the first two principal components
num_components = 2
components = eigenvectors[:, -num_components:]  # select the last two components
X_pca = torch.matmul(X_tensor, components)

# Convert PyTorch tensor back to numpy array for plotting
X_pca_np = X_pca.numpy()

# Plot 2D projection
plt.figure(figsize=(8, 6))
plt.scatter(X_pca_np[:, 0], X_pca_np[:, 1], c=y, alpha=0.5)
plt.title('PCA of Wine Quality Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Wine Quality')
plt.grid(True)
plt.show()
