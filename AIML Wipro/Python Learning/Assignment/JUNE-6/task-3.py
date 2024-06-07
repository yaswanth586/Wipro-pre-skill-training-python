import numpy as np

# a. Mathematical functions:
# Create an array of 10 evenly spaced values between 0 and 2π.
# Compute the sine, cosine, and tangent of each value.
# Print the results.

values = np.linspace(0, 2 * np.pi, 10)

sine_values = np.sin(values)
cosine_values = np.cos(values)
tangent_values = np.tan(values)

print("Values between 0 and 2π: ", values)
print("Sine of each value: ", sine_values)
print("Cosine of each value: ", cosine_values)
print("Tangent of each value: ", tangent_values)

# b. Statistical functions:
# Create a 3x3 array with random integers between 1 and 100.
# Compute the mean, median, standard deviation, and variance.
# Print the results.

array_3x3 = np.random.randint(1, 101, size=(3, 3))

mean_value = np.mean(array_3x3)
median_value = np.median(array_3x3)
std_deviation = np.std(array_3x3)
variance_value = np.var(array_3x3)

print("3x3 Array:\n", array_3x3)
print("Mean:", mean_value)
print("Median:", median_value)
print("Standard Deviation:", std_deviation)
print("Variance:", variance_value)
