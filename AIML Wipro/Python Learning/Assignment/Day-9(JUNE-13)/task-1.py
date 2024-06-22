# Implementing Linear Regression with PyTorch
# Problem 1: Load a dataset (such as the Boston Housing dataset) and
# implement a simple linear regression model to predict housing prices.
# Use Mean Squared Error (MSE) as the loss function.
# Problem 2: Plot the predicted vs actual values for the training and test sets.

from sklearn.datasets import fetch_california_housing
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Load the dataset
california = fetch_california_housing()
X = california.data
y = california.target

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # reshape to column vector


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


# Instantiate the model
model = LinearRegression(X.shape[1], 1)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

num_epochs = 1000

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
# Convert tensors to numpy arrays
X_np = X.detach().numpy()
y_np = y.detach().numpy()

# Prediction
model.eval()
with torch.no_grad():
    predicted = model(X).detach().numpy()

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(y_np, predicted, c='r', label='Predictions')
plt.plot([y_np.min(), y_np.max()], [y_np.min(), y_np.max()], 'b--', lw=2, label='Actual')
plt.title('Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
plt.show()
