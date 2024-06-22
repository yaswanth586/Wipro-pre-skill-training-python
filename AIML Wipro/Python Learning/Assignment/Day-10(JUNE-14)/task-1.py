# Implement logistic regression to classify the Balance scale dataset into two classes
# using only 'L', and 'R' using PyTorch.
# Evaluate the model's performance using accuracy.
# Steps:
# Load the Balance scale dataset.
# Preprocess the data: normalize the features and drop the 'B' value in target.
# Split the data into training and testing sets.
# Implement the logistic regression model using PyTorch.
# Train the model and evaluate its performance on the test set.
# Report the accuracy of the model.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

# Load the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data'
column_names = ['Class', 'Left-Weight', 'Left-Distance', 'Right-Weight', 'Right-Distance']
data = pd.read_csv(url, names=column_names)

# Preprocess the data: drop 'B' class and normalize features
data = data[data['Class'] != 'B']
X = data.drop('Class', axis=1).values
y = data['Class'].apply(lambda x: 0 if x == 'L' else 1).values  # Encode 'L' as 0 and 'R' as 1

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


# Define the logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


# Instantiate the model
model = LogisticRegression(X_train.shape[1], 1)

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
with torch.no_grad():
    model.eval()
    y_pred = model(X_test)
    y_pred_class = y_pred.round().numpy()
    accuracy = accuracy_score(y_test.numpy(), y_pred_class)
    print(f'Logistic Regression Test Accuracy: {accuracy:.4f}')
