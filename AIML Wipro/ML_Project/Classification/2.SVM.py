# SVM with PyTorch

import torch
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
import matplotlib.pyplot as plt

# Generate synthetic data
X, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, random_state=42)
y[y == 0] = -1  # Change labels from {0, 1} to {-1, 1}

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float().view(-1, 1)
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float().view(-1, 1)


class SVMModel(nn.Module):
    def __init__(self):
        super(SVMModel, self).__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


model = SVMModel()


# Define the loss function and the optimizer
def hinge_loss(outputs, labels):
    return torch.mean(torch.clamp(1 - outputs * labels, min=0))


optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1000
losses = []
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_train_tensor)
    loss = hinge_loss(outputs, y_train_tensor)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plot training loss
plt.plot(range(num_epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# Evaluating Model

# Evaluation
model.eval()
with torch.no_grad():
    y_pred_train = model(X_train_tensor).sign().view(-1)
    y_pred_test = model(X_test_tensor).sign().view(-1)

# Calculate accuracy
train_accuracy = (y_pred_train.eq(y_train_tensor.view(-1)).sum() / float(y_train_tensor.shape[0])).item()
test_accuracy = (y_pred_test.eq(y_test_tensor.view(-1)).sum() / float(y_test_tensor.shape[0])).item()

print(f'Train Accuracy: {train_accuracy * 100:.2f}%')
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')


# Plot decision boundary
def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    grid = np.c_[xx.ravel(), yy.ravel()]
    with torch.no_grad():
        grid_tensor = torch.from_numpy(grid).float()
        zz = model(grid_tensor).sign().numpy().reshape(xx.shape)
    plt.contourf(xx, yy, zz, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.show()


plot_decision_boundary(X_train, y_train, model)
plot_decision_boundary(X_test, y_test, model)


# ===============================================================



import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Function importing Dataset
def importdata():
        balance_data = pd.read_csv('balance-scale.data',
                                sep= ',', header = None)
        print ("Dataset Length: ", len(balance_data))
        print ("Dataset Shape: ", balance_data.shape)
        print ("Dataset")
        print(balance_data.head())
        return balance_data

def splitdataset(balance_data):
        X = balance_data.values[:, 1:5]
        Y = balance_data.values[:, 0]
        X_train, X_test, y_train, y_test = train_test_split(
                                        X, Y, test_size = 0.3, random_state = 100)
        return X, Y, X_train, X_test, y_train, y_test

def train_using_gini(X_train, X_test, y_train):
        clf_gini = DecisionTreeClassifier(criterion = "gini",
                random_state = 100,max_depth=3, min_samples_leaf=5)
        clf_gini.fit(X_train, y_train)
        return clf_gini

def prediction(X_test, clf_object):
        y_pred = clf_object.predict(X_test)
        print("Predicted values:")
        print(y_pred)
        return y_pred

def cal_accuracy(y_test, y_pred):
        print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
        print ("Accuracy : ",
        accuracy_score(y_test,y_pred)*100)
        print("Report : ",
        classification_report(y_test, y_pred))


#driver code
data = importdata()
X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
clf_gini = train_using_gini(X_train, X_test, y_train)
print("Results Using Gini Index:")
y_pred_gini = prediction(X_test, clf_gini)
cal_accuracy(y_test, y_pred_gini)