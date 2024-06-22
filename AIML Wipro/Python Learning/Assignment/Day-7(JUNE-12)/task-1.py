# Creating and Manipulating PyTorch Tensors
# Problem 1: Create a 3x3 tensor with random values and perform the following operations:
# Add 10 to each element.
# Multiply each element by 2.
# Calculate the mean and standard deviation of the tensor.
# Problem 2: Create two tensors of size 3x3 with random values and perform element-wise addition and multiplication.
# Problem 3: Create a tensor with values from 1 to 16 and reshape it to a 4x4 tensor.
# Extract the first two rows and last two columns.

import torch

# Problem 1: Create a 3x3 tensor with random values and perform operations
# Create a 3x3 tensor with random values
tensor1 = torch.rand(3, 3)
print("Original 3x3 Tensor:\n", tensor1)

# Add 10 to each element
tensor1_added = tensor1 + 10
print("\nTensor after adding 10 to each element:\n", tensor1_added)

# Multiply each element by 2
tensor1_multiplied = tensor1_added * 2
print("\nTensor after multiplying each element by 2:\n", tensor1_multiplied)

# Calculate the mean and standard deviation
mean = tensor1_multiplied.mean()
std_dev = tensor1_multiplied.std()

print("\nMean of the tensor:", mean.item())
print("Standard Deviation of the tensor:", std_dev.item())

# Problem 2: Create two tensors of size 3x3 with random values and perform element-wise addition and multiplication

# Create two 3x3 tensors with random values
tensor2 = torch.rand(3, 3)
tensor3 = torch.rand(3, 3)

print("\nFirst 3x3 Tensor:\n", tensor2)
print("Second 3x3 Tensor:\n", tensor3)

# Perform element-wise addition
tensor_add = tensor2 + tensor3
print("\nElement-wise Addition of tensors:\n", tensor_add)

# Perform element-wise multiplication
tensor_multiply = tensor2 * tensor3
print("\nElement-wise Multiplication of tensors:\n", tensor_multiply)

# Problem 3: Create a tensor with values from 1 to 16 and reshape it to a 4x4 tensor.
# Extract the first two rows and last two columns

# Create a tensor with values from 1 to 16
tensor4 = torch.arange(1, 17)
print("\nOriginal Tensor with values from 1 to 16:\n", tensor4)

# Reshape it to a 4x4 tensor
tensor4_reshaped = tensor4.view(4, 4)
print("\n4x4 Reshaped Tensor:\n", tensor4_reshaped)

# Extract the first two rows and last two columns
extracted_tensor = tensor4_reshaped[:2, 2:]
print("\nExtracted Tensor (first two rows, last two columns):\n", extracted_tensor)
