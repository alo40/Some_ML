"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    # Parameters
    # ---------------------------------------------------
    K = len(mixture.p)
    d = X.shape[1]
    n = X.shape[0]
    weight = mixture.p
    mean = mixture.mu
    cova = [var * np.identity(d) for var in mixture.var]

    posterior = np.zeros((n, K))
    multi_pdf = np.zeros((n, K))
    likelihood = np.zeros(n)

    # E-Step
    # ---------------------------------------------------
    # Calculate likelihood
    for i, x_i in enumerate(X):
        for j in range(K):
            norm_term = 1 / np.sqrt(np.linalg.det(cova[j]) * (2 * np.pi) ** d)
            expo_term = np.exp(-0.5 * (x_i - mean[j]).T @ np.linalg.inv(cova[j]) @ (x_i - mean[j]))
            multi_pdf[i, j] = norm_term * expo_term
            likelihood[i] += weight[j] * multi_pdf[i, j]  # sum over all K gaussian mixtures
    joint_log_likelihood = np.sum(np.log(likelihood))

    # Calculate posteriors
    for j in range(K):
        for i, x_i in enumerate(X):
            posterior[i, j] = weight[j] * multi_pdf[i, j] / likelihood[i]

    return posterior, joint_log_likelihood


def mstep(X: np.ndarray, posterior: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    # Parameters
    # ---------------------------------------------------
    d = X.shape[1]
    n = X.shape[0]
    K = posterior.shape[1]
    mean = np.zeros((K, d))
    var = np.zeros(K)
    weight = np.zeros(K)

    # M-step
    # ---------------------------------------------------
    for j in range(K):
        mean[j] = posterior[:, j] @ X / sum(posterior[:, j])
        weight[j] = sum(posterior[:, j]) / n
        var[j] = posterior[:, j] @ np.linalg.norm(X - mean[j], axis=1) ** 2 / (d * sum(posterior[:, j]))
    mixture = GaussianMixture(mean, var, weight)

    return mixture


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    i = 0
    old_joint_log_likelihood = 0
    while True:
        # EM algorithm
        # ---------------------------------------------------
        posterior, joint_log_likelihood = estep(X, mixture)
        mixture = mstep(X, posterior)

        # Convergence criteria
        # ---------------------------------------------------
        delta_log_likelihood = abs(joint_log_likelihood - old_joint_log_likelihood)
        if delta_log_likelihood <= 1e-6 * abs(joint_log_likelihood):
            return mixture, posterior, joint_log_likelihood
        else:
            old_joint_log_likelihood = joint_log_likelihood

        # Print results
        # ---------------------------------------------------
        # print(f"i={i}, LL_delta={delta_log_likelihood:.8f}")
        i += 1


## NOT USED -----------------------------------------------------------------------------

    # M-step (correct)
    # ----------------------------------
    # for j in range(K):
    #     for i, x_i in enumerate(X):
    #         mean[j] += posterior[i, j] * x_i
    #     mean[j] = mean[j] / sum(posterior[:, j])
    #
    #
    # for j in range(K):
    #     for i, x_i in enumerate(X):
    #         var[j] += posterior[i, j] * np.linalg.norm(x_i - mean[j]) ** 2
    #     var[j] = var[j] / (d * sum(posterior[:, j]))