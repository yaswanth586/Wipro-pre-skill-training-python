import numpy as np

# a. Array arithmetic:
# Create two 1-dimensional arrays of integers from 1 to 5 and 6 to 10.
# Perform element-wise addition, subtraction, multiplication, and division and Print the results.

array_1 = np.array([1, 2, 3, 4, 5])
array_2 = np.array([6, 7, 8, 9, 10])

addition = array_1 + array_2
subtraction = array_1 - array_2
multiplication = array_1 * array_2
division = array_1 / array_2

print("Element-wise addition:", addition)
print("Element-wise subtraction:", subtraction)
print("Element-wise multiplication:", multiplication)
print("Element-wise division:", division)

# b. Indexing and slicing:
# Create a 5x5 array with values from 1 to 25.
# Extract the subarray consisting of the first two rows and columns.
# Print the extracted subarray.

array_5x5 = np.arange(1, 26).reshape(5, 5)

sub_array = array_5x5[:2, :2]

print("Original 5x5 array:\n", array_5x5)
print("Extracted subarray (first two rows and columns):\n", sub_array)

# c. Boolean indexing:
# Create a 1-dimensional array of integers from 10 to 19.
# Extract elements greater than 15.
# Print the resulting array.

array_1d = np.arange(10, 20)

greater_than_15 = array_1d[array_1d > 15]

print("Original array: ", array_1d)
print("Elements greater than 15: ", greater_than_15)
