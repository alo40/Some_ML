"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
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
            Cu = np.where(x_i != 0)[0]
            cova_cu = np.diag(cova[j])[Cu] * np.identity(len(Cu))
            norm_term = 1 / np.sqrt(np.linalg.det(cova_cu) * (2 * np.pi) ** d)
            expo_term = np.exp(-0.5 * (x_i[Cu] - mean[j][Cu]).T @ np.linalg.inv(cova_cu) @ (x_i[Cu] - mean[j][Cu]))
            multi_pdf[i, j] = norm_term * expo_term
            likelihood[i] += weight[j] * multi_pdf[i, j]  # sum over all K gaussian mixtures
    joint_log_likelihood = np.sum(np.log(likelihood))

    # Calculate posteriors
    for j in range(K):
        for i, x_i in enumerate(X):
            posterior[i, j] = weight[j] * multi_pdf[i, j] / likelihood[i]

    return posterior, joint_log_likelihood



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    raise NotImplementedError


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
    raise NotImplementedError


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
