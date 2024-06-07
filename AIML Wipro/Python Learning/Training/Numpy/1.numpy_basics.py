"""
NumPy Arrays
Creating Arrays from Lists
"""

import numpy as np

# Creating a 1D array from a list
list_1d = [1, 2, 3, 4, 5]
array_1d = np.array(list_1d)
print("1D Array from list:", array_1d)

# Creating a 2D array from a list of lists
list_2d = [[1, 2, 3], [4, 5, 6]]
array_2d = np.array(list_2d)
print("2D Array from list of lists:\n", array_2d)


'''
Array Attributes
Shape and Size: Get the shape and size of a NumPy array using the shape
 and size attributes.
Data Types: NumPy arrays have a single data type for all elements. 
You can check the data type with the dtype attribute.
Specify the data type when creating an array.

'''

array = np.array([[1, 2, 3], [4, 5, 6]])

# Shape of the array
print("Shape of array:", array.shape)

# Size of the array
print("Size of array:", array.size)

# Data type of the array
print("Data type of array:", array.dtype)

array_float = np.array([1, 2, 3], dtype=np.float64)
print("Array with specified data type:", array_float)
print("Data type of the array:", array_float.dtype)


'''
Built-in Functions
arange: Similar to Python's range() but returns a NumPy array.
linspace: Creates an array of evenly spaced values over a specified range.
ones: Creates an array filled with ones.
zeros: Creates an array filled with zeros.
'''

array_arange = np.arange(0, 10, 2)
print("Array with arange:", array_arange)

array_linspace = np.linspace(10, 100, 5, dtype=np.int32)
print("Array with linspace:", array_linspace)

array_ones = np.ones((2, 3))
print("Array of ones:\n", array_ones)

array_zeros = np.zeros((2, 3))
print("Array of zeros:\n", array_zeros)
