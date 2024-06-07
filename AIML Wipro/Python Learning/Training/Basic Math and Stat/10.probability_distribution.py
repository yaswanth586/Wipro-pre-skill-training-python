import numpy as np
from scipy.stats import bernoulli, binom, poisson

#DISCRETE DISTRIBUTION
# Bernoulli Distribution
p = 0.5
bernoulli_rv = bernoulli(p)
print(f"Bernoulli PMF: P(X=1) = {bernoulli_rv.pmf(1)}, P(X=0) = {bernoulli_rv.pmf(0)}")

# Binomial Distribution
n = 10
p = 0.5
binom_rv = binom(n, p)
k = 5
print(f"Binomial PMF: P(X={k}) = {binom_rv.pmf(k)}")

# Poisson Distribution
lam = 3
poisson_rv = poisson(lam)
k = 4
print(f"Poisson PMF: P(X={k}) = {poisson_rv.pmf(k)}")


#CONTINUOUS DISTRIBUTION
from scipy.stats import uniform, norm, expon

# Uniform Distribution
a, b = 0, 1
uniform_rv = uniform(a, b-a)
print(f"Uniform PDF: f(x=0.5) = {uniform_rv.pdf(0.5)}")

# Normal Distribution
mu, sigma = 0, 1
norm_rv = norm(mu, sigma)
print(f"Normal PDF: f(x=0) = {norm_rv.pdf(0)}")


#PROP & VAL
from scipy.stats import norm

# Normal Distribution
mu, sigma = 0, 1
norm_rv = norm(mu, sigma)

# Mean
mean = norm_rv.mean()
print(f"Mean: {mean}")

# Variance
variance = norm_rv.var()
print(f"Variance: {variance}")

# Skewness
skewness = norm_rv.stats(moments='s')
print(f"Skewness: {skewness}")

# Kurtosis
kurtosis = norm_rv.stats(moments='k')
print(f"Kurtosis: {kurtosis}")