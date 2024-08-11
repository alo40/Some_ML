import numpy as np


def gaussian_1d(x, mu, sigma):
    coeff = 1 / (np.sqrt(2 * np.pi * sigma ** 2))
    exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
    gaussian = coeff * np.exp(exponent)
    return gaussian


def gaussian_2d(X, Y, mu_x, mu_y, sigma_x, sigma_y, rho):
    coeff = 1 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho**2))
    exponent = - (1 / (2 * (1 - rho**2))) * (
        ((X - mu_x)**2 / sigma_x**2) +
        ((Y - mu_y)**2 / sigma_y**2) -
        (2 * rho * (X - mu_x) * (Y - mu_y)) / (sigma_x * sigma_y)
    )
    gaussian = coeff * np.exp(exponent)
    return gaussian


def compute_x(clusters):
    x = np.append(clusters[0], clusters[1], axis=0)
    return x


def compute_delta_j(x, K):
    delta = np.zeros((K, x.shape[0]))
    # for j in range(K):
        # for i, x_i in enumerate(x):  # needs to be refined using clustering methods
        #     if j == 0 and i < 3:
        #         delta[j, i] = 1
        #     if j == 1 and i >= 3:
        #         delta[j, i] = 1
        # print(f"delta[{j}, {i}] = {delta[j, i]}")

    delta[0] = np.random.choice([0, 1], size=x.shape[0])
    delta[1] = 1 - delta[0]

    return delta


def compute_n_j(delta, K):
    n = np.zeros(K)
    for j in range(K):
        n[j] = sum(delta[j])
    return n


def compute_mu_j(x, delta, n, K):
    mu = np.zeros((K, x.shape[1]))
    for j in range(K):
        sum = 0
        for i, x_i in enumerate(x):
           sum += delta[j, i] * x_i
        mu[j] = sum / n[j]
    return mu


def compute_sigma_j(x, delta, n, mu, K):
    variance = np.zeros(K)
    for j in range(K):
        sum = 0
        for i, x_i in enumerate(x):
            sum += delta[j, i] * np.linalg.norm(x_i - mu[j]) ** 2
        variance[j] = sum / (n[j] * x.shape[1])
    sigma = np.sqrt(variance)
    return sigma
