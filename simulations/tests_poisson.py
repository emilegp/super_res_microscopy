import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt

# Parameters
lambda_ = 5  # average number of events
size = 1000  # number of samples

# Generate random samples from a Poisson distribution
poisson_samples = poisson.rvs(mu=lambda_, size=size)

# Plot the histogram of samples
plt.hist(poisson_samples, bins=30, density=True, alpha=0.6, color='g')

# Overlay the theoretical PMF
x = np.arange(0, max(poisson_samples))
pmf = poisson.pmf(x, mu=lambda_)
plt.plot(x, pmf, 'bo', ms=8, label="Poisson PMF")
plt.legend()
plt.xlabel('Value')
plt.ylabel('Probability')
plt.title('Poisson Distribution (lambda = 5)')
plt.show()
