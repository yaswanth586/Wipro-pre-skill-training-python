# Apply t-SNE for Dimensionality Reduction and Visualize High-Dimensional Data in 2D or 3D
# Dataset: Wine Quality Dataset
# Objective: Apply t-SNE on the Wine Quality dataset to reduce its dimensionality and visualize the results in 2D or 3D.
# Steps:
# Load the Wine Quality dataset:
# Use pandas to load the Wine Quality dataset from the UCI Machine Learning Repository.
# Preprocess the data:
# Normalize the features using StandardScaler.
# Apply t-SNE:
# Use sklearn.manifold.TSNE to reduce the data to 2 or 3 dimensions.
# Visualize the Results:
# Use matplotlib to plot the 2D or 3D projection of the data.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
wine_data = pd.read_csv(url, sep=';')

# Separate features and labels
X = wine_data.drop(columns=['quality'])  # features
y = wine_data['quality']  # labels (not used in t-SNE)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled)

# Apply t-SNE to reduce to 2 or 3 dimensions
# Let's reduce to 2 dimensions for visualization
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

print(X_tsne)

# Visualize the Results: 2D Projection
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.5)
plt.title('t-SNE Visualization of Wine Quality Dataset')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(label='Wine Quality')
plt.grid(True)
plt.show()
