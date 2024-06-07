import numpy as np

# a. Create a 1-dimensional array:
# Create a 1-dimensional array of integers from 0 to 9.
# Print the array and its shape.

array_1d = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print("1Dimensional Array: ", array_1d)
print("Shape of the 1-dimensional array: ", array_1d.shape)

# b. Create a 2-dimensional array:
# Create a 2-dimensional array (3x3) with values from 1 to 9.
# Print the array, its shape, and the sum of all elements.

array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("2Dimensional Array: ", array_2d)
print("Shape of the 2-dimensional array: ", array_2d.shape)

sum_elements = np.sum(array_2d)
print("Sum of all elements in 2-dimensional array: ", sum_elements)

# c. Reshape the array:
# Reshape the 1-dimensional array from step 1 into a 2x5 array.
# Print the reshaped array and its shape.

print("Array: ", array_1d)
print("shape of the array: ", array_1d.shape)

array_reshape_2d = array_1d.reshape(2, 5)

print("Array after reshape:\n", array_reshape_2d)
print("Shape of the array after reshape: ", array_reshape_2d.shape)

