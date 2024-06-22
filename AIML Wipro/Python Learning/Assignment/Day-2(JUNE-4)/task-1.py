# Matrix Operations with NumPy
#
# Create two 3x3 matrices A and B with random integer values between 1 and 10.
# Compute the following:
# The sum of A and B.
# The difference between A and B.
# The element-wise product of A and B.
# The matrix product of A and B.
# The transpose of matrix A.
# The determinant of matrix A.

import numpy as np

# random matrices A and B between 1 and 10
A = np.random.randint(1, 10, size=(3, 3))
B = np.random.randint(1, 10, size=(3, 3))

print("Matrix A: ", A)
print("Matrix B: ", B)

# sum of A and B
sum_result = A + B
print("Sum of A and B: ", sum_result)

# difference between A and B
diff_result = A - B
print("Difference between A and B: ", diff_result)

# element wise product of A and B
elementwise_product_result = A * B
print("Element-wise product of A and B: ", elementwise_product_result)

# matrix product of A and B
matrix_product_result = np.dot(A, B)
print("\nMatrix product of A and B:")
print(matrix_product_result)

# transpose of matrix A
transpose_A = np.transpose(A)
print("\nTranspose of matrix A:")
print(transpose_A)

# determinant of matrix A
determinant_A = np.linalg.det(A)
print("\nDeterminant of matrix A:", determinant_A)