# Performing Polynomial Regression and Feature Engineering
# Problem 1: Load a dataset and perform polynomial regression.
# Add polynomial features up to degree 3 and train the model using these features.
# Problem 2: Compare the performance of the polynomial regression model with the linear regression model using MSE.


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Load the dataset
california = fetch_california_housing()
X = california.data
y = california.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # reshape to column vector
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Add polynomial features up to degree 3
poly = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly = poly.fit_transform(X_train.numpy())
X_test_poly = poly.transform(X_test.numpy())

# Convert back to PyTorch tensors
X_train_poly = torch.tensor(X_train_poly, dtype=torch.float32)
X_test_poly = torch.tensor(X_test_poly, dtype=torch.float32)

class PolynomialRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolynomialRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

# Instantiate the polynomial regression model
model_poly = PolynomialRegression(X_train_poly.shape[1], 1)

# Define loss function and optimizer for polynomial regression
criterion = nn.MSELoss()
optimizer = optim.Adam(model_poly.parameters(), lr=0.001)

# Training loop for polynomial regression
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model_poly(X_train_poly)
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Polynomial Regression Loss: {loss.item():.4f}')

# Instantiate the linear regression model
model_linear = nn.Linear(X_train.shape[1], 1)

# Define loss function and optimizer for linear regression
criterion_linear = nn.MSELoss()
optimizer_linear = optim.Adam(model_linear.parameters(), lr=0.001)

# Training loop for linear regression
for epoch in range(num_epochs):
    # Forward pass
    outputs = model_linear(X_train)
    loss = criterion_linear(outputs, y_train)

    # Backward pass and optimization
    optimizer_linear.zero_grad()
    loss.backward()
    optimizer_linear.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Linear Regression Loss: {loss.item():.4f}')

# Evaluation on test set
with torch.no_grad():
    model_poly.eval()
    y_pred_poly = model_poly(X_test_poly)
    mse_poly = mean_squared_error(y_test.numpy(), y_pred_poly.numpy())
    print(f'Polynomial Regression Test MSE: {mse_poly:.4f}')

    model_linear.eval()
    y_pred_linear = model_linear(X_test)
    mse_linear = mean_squared_error(y_test.numpy(), y_pred_linear.numpy())
    print(f'Linear Regression Test MSE: {mse_linear:.4f}')

