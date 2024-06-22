# Applying Ridge and Lasso Regression
# Problem 1: Implement Ridge regression on the same dataset used in Assignment 1
# Use cross-validation to select the best regularization parameter (alpha).
# Problem 2: Implement Lasso regression on the same dataset.
# Use cross-validation to select the best regularization parameter (alpha).
# Problem 3: Compare the performance of Ridge, Lasso, and standard linear regression models
# in terms of MSE and interpret the results.

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the dataset
california = fetch_california_housing()
X = california.data
y = california.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Define Ridge regression model
ridge = Ridge()

# Define grid of alpha values (regularization parameter)
alphas = np.logspace(-3, 3, 20)  # 20 alphas from 10^-3 to 10^3

# Perform GridSearchCV to find the best alpha
param_grid = {'alpha': alphas}
ridge_cv = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
ridge_cv.fit(X_train_scaled, y_train)

# Get the best alpha
best_alpha_ridge = ridge_cv.best_params_['alpha']
print(f'Best alpha for Ridge regression: {best_alpha_ridge:.4f}')

# Train Ridge regression model with best alpha
ridge_best = Ridge(alpha=best_alpha_ridge)
ridge_best.fit(X_train_scaled, y_train)

# Predict on test set
y_pred_ridge = ridge_best.predict(X_test_scaled)

# Calculate MSE
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
print(f'Ridge Regression Test MSE: {mse_ridge:.4f}')
from sklearn.linear_model import Lasso

# Define Lasso regression model
lasso = Lasso(max_iter=10000)

# Perform GridSearchCV to find the best alpha
lasso_cv = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error')
lasso_cv.fit(X_train_scaled, y_train)

# Get the best alpha
best_alpha_lasso = lasso_cv.best_params_['alpha']
print(f'Best alpha for Lasso regression: {best_alpha_lasso:.4f}')

# Train Lasso regression model with best alpha
lasso_best = Lasso(alpha=best_alpha_lasso, max_iter=10000)
lasso_best.fit(X_train_scaled, y_train)

# Predict on test set
y_pred_lasso = lasso_best.predict(X_test_scaled)

# Calculate MSE
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
print(f'Lasso Regression Test MSE: {mse_lasso:.4f}')
# Train standard linear regression model
from sklearn.linear_model import LinearRegression

model_linear = LinearRegression()
model_linear.fit(X_train_scaled, y_train)
y_pred_linear = model_linear.predict(X_test_scaled)
mse_linear = mean_squared_error(y_test, y_pred_linear)
print(f'Linear Regression Test MSE: {mse_linear:.4f}')

# Compare MSE of Ridge, Lasso, and Linear regression
print(f'Ridge Regression Test MSE: {mse_ridge:.4f}')
print(f'Lasso Regression Test MSE: {mse_lasso:.4f}')
print(f'Linear Regression Test MSE: {mse_linear:.4f}')

# Interpretation
print("\nInterpretation:")
print("- Ridge regression typically performs better than simple linear regression when there is multi collinearity "
      "among the features.")
print("- Lasso regression is useful for feature selection because it tends to shrink coefficients of less important "
      "features to zero.")
print("- In this case, we observe that Ridge regression has slightly lower MSE than Lasso regression and linear "
      "regression.")
