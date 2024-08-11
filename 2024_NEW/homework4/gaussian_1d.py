import numpy as np
import matplotlib.pyplot as plt
from gaussian_function_dict import *

# ------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------

K = 2  # number of clusters
T = 30  # total EM iterations

# generate exercise data
x = np.array([-1., 0., 4., 5., 6.])
y = np.zeros_like(x)

# Parameters for the Gaussian function
mu = np.array([6., 7.])
sigma = np.array([np.sqrt(1.), 2.])
p = np.array([0.5, 0.5])
posterior = np.zeros((K, len(x)))
likelihood = np.zeros((K, len(x)))

# ------------------------------------------------------------------------
# Testing
# ------------------------------------------------------------------------

# gauss_00 = np.exp(-(x[0] - mu[0]) ** 2 / (2 * sigma[0] ** 2)) / np.sqrt(2 * np.pi * sigma[0] ** 2)
# gauss_01 = np.exp(-(x[0] - mu[1]) ** 2 / (2 * sigma[1] ** 2)) / np.sqrt(2 * np.pi * sigma[1] ** 2)
# post_00 = gauss_00 * p[0] / (gauss_00 * p[0] + gauss_01 * p[1])
# print(post_00)

# ------------------------------------------------------------------------
# EM algo
# ------------------------------------------------------------------------

likelihood_vector = []
for t in range(T):

    # E-step
    # ----------------------------------
    # calculate likelihood
    likelihood = np.zeros(len(x))
    for j in range(K):
        likelihood += np.dot(p[j], gaussian_1d(x, mu[j], sigma[j]))
    likelihood_vector.append(np.prod(likelihood))

    # Calculate posteriors
    for j in range(K):
        for i, x_i in enumerate(x):
            posterior[j, i] = p[j] * gaussian_1d(x_i, mu[j], sigma[j]) / likelihood[i]

    # M-step
    # ----------------------------------
    d = 1  # number of dimensions in x
    for j in range(K):
        mu[j] = np.dot(posterior[j], x) / sum(posterior[j])
        p[j] = sum(posterior[j]) / len(x)
        sigma[j] = np.dot(posterior[j], (x - mu[j]) ** 2) / (d * sum(posterior[j]))

    print(f"Iteration: {t}")
    print(f"Posteriors: \n{posterior}")
    print(f"Likelihood: {np.prod(likelihood)}")
    print(f"Means: {mu}")
    print(f"Standard Deviations: {sigma}")
    print(f"Mixing Coefficients: {p}\n")

# Plot likelihood convergence
plt.plot(likelihood_vector)
plt.show()

# ------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------

# Generate x values
lim = 10
x_ = np.linspace(-lim, lim, 1000)

# Compute the Gaussian function for plotting
gaussian1 = gaussian_1d(x_, mu[0], sigma[0])
gaussian2 = gaussian_1d(x_, mu[1], sigma[1])

# Plot the Gaussian function
plt.plot(x_, gaussian1, label=f'Gaussian1 $\mu={mu[0]:.5f}$, $\sigma={sigma[0]:.5f}$', color='blue')
plt.plot(x_, gaussian2, label=f'Gaussian2 $\mu={mu[1]:.5f}$, $\sigma={sigma[1]:.5f}$', color='red')
plt.scatter(x, y, color='blue', s=100)
plt.title('1D Gaussian Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
# plt.ylim(0, 1)
plt.grid(True)
plt.show()
