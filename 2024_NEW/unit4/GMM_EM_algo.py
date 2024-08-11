import numpy as np
from scipy.stats import norm


def initialize_parameters(X, k):
    n = X.shape[0]
    # mu = np.random.choice(X, k, replace=False)
    mu = np.array([-3., 2.])
    # sigma = np.ones(k)  # Initial standard deviations
    sigma = np.array([4., 4.])
    pi = np.ones(k) / k  # Initial mixing coefficients
    return mu, sigma, pi


def e_step(X, mu, sigma, pi):
    k = len(pi)
    n = X.shape[0]
    responsibilities = np.zeros((n, k))

    for i in range(k):
        responsibilities[:, i] = pi[i] * norm.pdf(X, mu[i], sigma[i])

    responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=True)
    return responsibilities


def m_step(X, responsibilities):
    n = X.shape[0]
    k = responsibilities.shape[1]
    Nk = responsibilities.sum(axis=0)

    mu = np.dot(responsibilities.T, X) / Nk
    sigma = np.zeros(k)
    pi = Nk / n

    for i in range(k):
        diff = X - mu[i]
        sigma[i] = np.sqrt(np.dot(responsibilities[:, i] * diff ** 2, np.ones(n)) / Nk[i])

    return mu, sigma, pi


def gmm_em(X, k, max_iter=100, tol=1e-6):
    mu, sigma, pi = initialize_parameters(X, k)

    for iteration in range(max_iter):
        responsibilities = e_step(X, mu, sigma, pi)
        mu_new, sigma_new, pi_new = m_step(X, responsibilities)

        # Check for convergence
        if np.allclose(mu, mu_new, atol=tol) and np.allclose(sigma, sigma_new, atol=tol):
            break

        mu, sigma, pi = mu_new, sigma_new, pi_new

    return mu, sigma, pi


# Example usage
# np.random.seed(0)
# X = np.random.randn(300)  # Example 1D data: 300 points
X = np.array([0.2, -0.9, -1, 1.2, 1.8])

k = 2  # Number of Gaussian components
mu, sigma, pi = gmm_em(X, k)

print("Means:\n", mu)
print("Standard Deviations:\n", sigma)
print("Mixing Coefficients:\n", pi)
