import numpy as np
import time
import numba

# a. Vectorization:
# Create a function to compute the element-wise square of an array using a for loop.
# Create another function to perform the same computation using NumPy vectorization.
# Compare the performance of the two functions using a large array of size 1,000,000.


def elementwise_square_loop(arr):
    result = np.zeros(len(arr))
    for i in range(len(arr)):
        result[i] = arr[i] ** 2
    return result


def elementwise_square_vectorized(arr):
    return np.square(arr)


large_array = np.random.rand(1000000)

start_time_loop = time.time()
squared_array_loop = elementwise_square_loop(large_array)
end_time_loop = time.time()

start_time_vectorized = time.time()
squared_array_vectorized = elementwise_square_vectorized(large_array)
end_time_vectorized = time.time()

print("Time taken using for loop:", end_time_loop - start_time_loop, "seconds")
print("Time taken using NumPy vectorization:", end_time_vectorized - start_time_vectorized, "seconds")


# b. Numba:
# Use the @numba.jit decorator to optimize the function from step 1 that uses a for loop.
# Compare the performance of the Numba-optimized function with the vectorized NumPy function.


@numba.jit(nopython=True)
def elementwise_square_numba(arr):
    result = np.zeros(len(arr))
    for i in range(len(arr)):
        result[i] = arr[i] ** 2
    return result


start_time_numba = time.time()
squared_array_numba = elementwise_square_numba(large_array)
end_time_numba = time.time()

print("Time taken using Numba optimization:", end_time_numba - start_time_numba, "seconds")
