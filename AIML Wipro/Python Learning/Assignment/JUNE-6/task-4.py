import numpy as np
import time

# a. Broadcasting:
# Create a 3x1 array with values from 1 to 3.
# Create a 1x3 array with values from 4 to 6.
# Add the two arrays using broadcasting.
# Print the resulting array.

array_3x1 = np.array([[1], [2], [3]])
array_1x3 = np.array([[4, 5, 6]])

result_array = array_3x1 + array_1x3

print("3x1 Array:\n", array_3x1)
print("1x3 Array:\n", array_1x3)
print("Resulting Array after Broadcasting Addition: \n", result_array)

# b. Vectorized operations:
# Create two large arrays of size 1,000,000 with random values.
# Compute the element-wise product of the two arrays.
# Print the time taken for the computation using vectorized operations.

array1 = np.random.rand(1000000)
array2 = np.random.rand(1000000)

start_time = time.time()
product = np.multiply(array1, array2)
end_time = time.time()

print("Time taken for element-wise product using vectorized operations:", end_time - start_time, "seconds")
