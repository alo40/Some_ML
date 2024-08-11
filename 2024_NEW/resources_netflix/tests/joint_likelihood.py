import numpy as np
from scipy.stats import multivariate_normal

# Define the GMM parameters
means = np.array([[0, 0, 0], [3, 3, 3], [3, 3, 3]])  # Means of the 2D Gaussian components
covariances = [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
               np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
               np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])]  # Covariance matrices
weights = np.array([0.4, 0.6, 0.4])  # Mixing coefficients

# Define multiple data points
data_points = np.array([
    [-1.636, 2.413, 1],
    [-2.957, 2.296, 1],
    [-2.871, 1.832, 1],
    [-1.636, 2.413, 1],
    [-2.957, 2.296, 1],
    [-2.871, 1.832, 1]
])

# Function to compute the likelihood of all data points
def compute_joint_likelihood(data_points, means, covariances, weights):
    n_points = data_points.shape[0]
    likelihoods = np.ones(n_points)

    for i, point in enumerate(data_points):
        # Compute PDF of each Gaussian component for the current data point
        pdfs = np.array([multivariate_normal.pdf(point, mean=mean, cov=cov)
                         for mean, cov in zip(means, covariances)])
        # Compute the likelihood for this data point
        likelihoods[i] = np.sum(weights * pdfs)

    # Compute the joint likelihood as the product of the individual likelihoods
    joint_likelihood = np.prod(likelihoods)

    return joint_likelihood


# Compute the joint likelihood for all data points
joint_likelihood = compute_joint_likelihood(data_points, means, covariances, weights)

print(f"Joint likelihood of the data points under the GMM: {joint_likelihood}")
