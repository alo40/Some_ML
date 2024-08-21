"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture
from scipy.stats import multivariate_normal
from tqdm import tqdm
from timeit import default_timer as timer
import warnings
warnings.filterwarnings('ignore')


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
    n, d = X.shape
    weight = mixture.p
    mean = mixture.mu
    # cova = [var * np.identity(d) for var in mixture.var]
    # posterior = np.zeros((n, K))
    posterior_numerical = np.zeros((n, K))
    multi_pdf = np.zeros((n, K))
    # likelihood = np.zeros(n)
    log_likelihood_numerical = np.zeros(n)
    a = np.zeros(K)  # temporal parameter used for logsumexp

    # E-Step
    # ---------------------------------------------------
    # Calculate likelihood
    # for i, x_i in enumerate(tqdm(X, desc="E-Step Likelihood", disable=True)):
    for i, x_i in enumerate(X):
        for j in range(K):
            # non-zero entries
            Cu = np.where(x_i != 0)[0]
            x_cu = x_i[Cu]
            d_cu = len(x_cu)  # very important term!
            mean_cu = mean[j][Cu]
            # cova_cu = np.diag(cova[j])[Cu] * np.identity(d_cu)

            # multi gaussian (self-made)
            cova_cu_det = np.power(mixture.var[j], d_cu)
            cova_cu_inv = (1 / mixture.var[j]) * np.identity(d_cu)
            try:
                norm_term = 1 / np.sqrt(cova_cu_det * np.power(2 * np.pi, d_cu))
            except OverflowError as e:
                norm_term = 0
            expo_term = np.exp(-0.5 * (x_cu - mean_cu).T @ cova_cu_inv @ (x_cu - mean_cu))
            multi_pdf[i, j] = norm_term * expo_term

            # term for likelihood using logsumexp
            a[j] = np.log(weight[j] + 1e-16) + np.log(multi_pdf[i, j] + 1e-16)

            # # not considering numerical underflow
            # likelihood[i] += weight[j] * multi_pdf[i, j]

        # to avoid numerical underflow
        log_likelihood_numerical[i] = logsumexp(a)

    # Calculate posteriors
    # for j in tqdm(range(K), desc="E-Step Posterior", disable=True):
    for j in range(K):
        for i, x_i in enumerate(X):
            # # not considering numerical underflow
            # posterior[i, j] = weight[j] * multi_pdf[i, j] / likelihood[i]

            # to avoid numerical underflow
            log_posterior = np.log(weight[j] + 1e-16) + np.log(multi_pdf[i, j] + 1e-16) - log_likelihood_numerical[i]
            posterior_numerical[i, j] = np.exp(log_posterior)

    # Calculate joint log likelihood
    # joint_log_likelihood = np.sum(np.log(likelihood))
    joint_log_likelihood_numerical = np.sum(log_likelihood_numerical)

    return posterior_numerical, joint_log_likelihood_numerical


def mstep(X: np.ndarray, posterior: np.ndarray, mixture: GaussianMixture, min_var: float = .25) -> GaussianMixture:
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
    # Parameters
    # ---------------------------------------------------
    n, d = X.shape
    K = posterior.shape[1]
    mean = np.zeros((K, d))
    mean_test = np.zeros((K, d))
    var = np.zeros(K)
    weight = np.zeros(K)
    # indicator_matrix = np.zeros_like(X)
    # upper_term_vector = np.zeros(d)
    # lower_term_vector = np.zeros(d)

    # M-step
    # ---------------------------------------------------
    # # Update mean (using loops)
    # for j in range(K):
    #     for l in tqdm(range(d), desc=f"M-Step Mean {j}", disable=True):
    #         upper_term = 0
    #         lower_term = 0
    #         for i, x_i in enumerate(X):
    #             # non-zero entries
    #             Cu = np.where(x_i != 0)[0]
    #             indicator = (np.isin(l, Cu) * 1)
    #             # indicator_matrix[i ,l] = indicator  # for testing
    #
    #             # mean terms
    #             upper_term += posterior[i, j] * indicator * x_i[l]
    #             lower_term += posterior[i, j] * indicator
    #
    #         # # save terms for later comparison
    #         # upper_term_vector[l] = upper_term  # for testing
    #         # lower_term_vector[l] = lower_term  # for testing
    #
    #         # to avoid erratic results update only if, otherwise use old value
    #         if lower_term >= 1:
    #             mean[j, l] = upper_term / lower_term
    #         else:
    #             mean[j, l] = mixture.mu[j, l]

    # Update mean (matrix)
    indicator_test = np.where(X != 0, 1, 0)
    for j in range(K):
        # upper_term_test = posterior[:, j] @ np.multiply(X, indicator_test)
        # lower_term_test = (posterior[:, j] @ indicator_test)
        mean[j] = posterior[:, j] @ np.multiply(X, indicator_test) / (posterior[:, j] @ indicator_test)

    # Update variance and mixture weight
    for j in tqdm(range(K), desc=f"M-Step Var", disable=True):
        upper_term = 0
        lower_term = 0
        for i, x_i in enumerate(X):
            # non-zero entries
            Cu = np.where(x_i != 0)[0]
            x_cu = x_i[Cu]
            d_cu = len(x_cu)  # very important term!
            mean_cu = mean[j][Cu]

            # var terms
            upper_term += posterior[i, j] * np.linalg.norm(x_cu - mean_cu) ** 2
            lower_term += posterior[i, j] * d_cu
            # print(f"M-Step update var i: {i}/{n}, j: {j}/{K}")
        var[j] = max(upper_term / lower_term, min_var)
        weight[j] = sum(posterior[:, j]) / n

    mixture = GaussianMixture(mean, var, weight)

    return mixture


def run(X: np.ndarray, mixture: GaussianMixture, post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
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
    # for i in tqdm(range(100), desc=f"Run"):
        # EM algorithm
        # ---------------------------------------------------
        # print(f"RUN Loop {i}")
        posterior, joint_log_likelihood = estep(X, mixture)
        mixture = mstep(X, posterior, mixture, min_var=.25)

        # Convergence criteria
        # ---------------------------------------------------
        delta_log_likelihood = abs(joint_log_likelihood - old_joint_log_likelihood)
        if delta_log_likelihood <= 1e-6 * abs(joint_log_likelihood):
            return mixture, posterior, joint_log_likelihood
        else:
            old_joint_log_likelihood = joint_log_likelihood

        # Print results
        # ---------------------------------------------------
        i += 1
        # print(f"i={i}, LL_delta={delta_log_likelihood:.8f}")


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    posterior, joint_log_likelihood = estep(X, mixture)
    K = posterior.shape[1]
    X_predict = X.copy()
    for i, x_row in enumerate(X):
        for j, x_j in enumerate(x_row):
             if x_j == 0:
                 for k in range(K):
                    X_predict[i, j] += posterior[i, k] * mixture.mu[k, j]
    return X_predict
