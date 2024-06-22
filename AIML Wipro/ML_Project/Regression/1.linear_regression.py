from statistics import mean

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style


def calculate_slope_intercept(xvalues, yvalues):
    m = (((mean(xvalues) * mean(yvalues)) - mean(xvalues * yvalues)) /
         ((mean(xvalues) * mean(xvalues)) - mean(xvalues * xvalues)))
    b = mean(yvalues) - m * mean(xvalues)
    return m, b


def linear_regression():
    regression_line = [(m * x) + b for x in xvalues]
    style.use('ggplot')
    plt.title('Training Data & Regression Line')
    plt.scatter(xvalues, yvalues, color='#003F72', label='Training Data')
    plt.plot(xvalues, regression_line, label='Reg Line')
    plt.legend(loc='best')
    plt.show()


def test_data():
    predict_xvalue = 7
    predict_yvalue = (m * predict_xvalue) + b
    print('Test Data for x :     ', predict_xvalue, '    ', 'Test Data for y :     ', predict_yvalue)
    plt.title('Train & Test Value')
    plt.scatter(xvalues, yvalues, color='#003F72', label='data')
    plt.scatter(predict_xvalue, predict_yvalue, color='#ff0000', label='Predicted Value')
    plt.legend(loc='best')
    plt.show()


def validate_results():
    predict_xvalues = np.array([2.5, 3.5, 4.5, 5.5, 6.5], dtype=np.float64)
    predict_yvalues = [(m * x) + b for x in predict_xvalues]
    print('Validation Data Set')
    print('X values', predict_xvalues)
    print('Y values', predict_yvalues)


# driver
xvalues = np.array([1, 2, 3, 4, 5], dtype=np.int32)
yvalues = np.array([14, 24, 34, 44, 54], dtype=np.int32)
m, b = calculate_slope_intercept(xvalues, yvalues)
print('Slope :  ', m, 'Intercept :  ', b)
linear_regression()
test_data()
validate_results()



import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

#Preparing the Dataset
# Generate some example data
np.random.seed(42)
X = np.random.rand(100, 1)
y = 5 * X + 2

# Convert to PyTorch tensors
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).float()

# Defining the Model

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()


# Training the Model

# Define the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluating the Model

model.eval()
with torch.no_grad():
    predicted = model(X_tensor).detach().numpy()

# Plotting the results
plt.scatter(X, y, label='Original data')
plt.plot(X, predicted, label='Fitted line', color='red')
plt.legend()
plt.show()
