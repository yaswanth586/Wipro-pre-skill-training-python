# Solving Linear Equations with SciPy
#
# Given the system of equations:
# 2x + 3y = 8
# 3x + 4y = 11
#
# Represent the system of equations in matrix form AX = B.
# Use scipy.linalg.solve to find the values of x and y.


import numpy as np
from scipy.linalg import solve

# define the coefficients matrix A
A = np.array([[2, 3],
              [3, 4]])

# define the constants matrix B
B = np.array([[8],
              [11]])

# solve the system of linear equations
inverse_a = np.linalg.inv(A)
solution = solve(A, B)

# extract the values of x and y from the solution
x = solution[0][0]
y = solution[1][0]

print("Solution:")
print("x =", x)
print("y =", y)
