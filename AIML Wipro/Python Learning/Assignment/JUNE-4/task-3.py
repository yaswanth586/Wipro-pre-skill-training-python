from scipy.integrate import quad
from scipy.misc import derivative


# define the function f(x) = x^3 + 2x^2 + x + 1
def f(x):
    return x**3 + 2*x**2 + x + 1


# compute the first derivative of f(x) at x = 1
first_derivative_at_1 = derivative(f, 1.0, n=1)

# compute the second derivative of f(x) at x = 1
second_derivative_at_1 = derivative(f, 1.0, n=2)

# compute the definite integral of f(x) from x = 0 to x = 2
definite_integral, _ = quad(f, 0, 2)

print("First derivative of f(x) at x = 1:", first_derivative_at_1)
print("Second derivative of f(x) at x = 1:", second_derivative_at_1)
print("Definite integral of f(x) from x = 0 to x = 2:", definite_integral)
