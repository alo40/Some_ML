import numpy as np
from scipy.stats import multivariate_normal

# Define the mean vector and covariance matrix
mean = np.array([0, 0])
cova = np.array([[1, 0.5], [0.5, 1]])  # covariance matrix

# Define multiple data points
X = np.array([
    [1, 1],
    [0, 0],
    [-1, -1]
])

# Compute the PDF for each data point (manually)
pdf = np.zeros(X.shape[0])
d = X.shape[1]
for i, x_i in enumerate(X):
    norm_term = 1 / np.sqrt(np.linalg.det(cova) * (2 * np.pi) ** d)
    expo_term = np.exp(-0.5 * (x_i - mean).T @ np.linalg.inv(cova) @ (x_i - mean))
    pdf[i] = norm_term * expo_term

# Compute the PDF for each data point
pdf_values = multivariate_normal.pdf(X, mean=mean, cov=cova)

print("PDF values for the data points:")
for point, pdf in zip(X, pdf_values):
    print(f"Data point {point} has PDF value {pdf}")
