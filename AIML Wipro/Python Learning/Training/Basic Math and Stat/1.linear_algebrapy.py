import numpy as np

# vector definition and notation
vector_2d = np.array([1, 2])
vector_3d = np.array([1, 2, 3])

print("Definition and notation of vectors using python")
print("2D Vector: ", vector_2d)
print("3D Vector: ", vector_3d)

# vector addition and subtraction
vector_a = np.array([1, 2])
vector_b = np.array([3, 4])

print(type(vector_a))

vector_add = vector_a + vector_b
vector_sub = vector_a - vector_b

print("Vector Addition: ", vector_add)
print("Subtraction: ", vector_sub)

# scalar multiplication with vector using python
scalar = 3
vector_scaled = scalar * vector_a

print("Scalar multiplication using python")
print("Scalar: ", scalar)
print("Vector: ", vector_a)
print("Scalar Multiplication: ", vector_scaled)

# dot product
dot_product = np.dot(vector_a, vector_b)

# magnitude of vector
magnitude_a = np.linalg.norm(vector_a)
magnitude_b = np.linalg.norm(vector_b)

# cosine of the angle between vectors
cos_theta = dot_product / (magnitude_a * magnitude_b)

print("Dot product and its properties using python")
print("Vector A: ", vector_a)
print("Vector B: ", vector_b)
print("Dot product (A . B): ", dot_product)
print("Magnitude of A: ", magnitude_a)
print("Magnitude of B: ", magnitude_b)
print("Cosine of the angle between vectors A and B: ", cos_theta)