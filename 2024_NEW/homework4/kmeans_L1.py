import numpy as np
from sklearn.metrics.pairwise import manhattan_distances


def kmeans_l1(X, k, max_iters=100, tol=1e-4):
    # # Randomly initialize cluster centers
    # np.random.seed(0)
    # centers = X[np.random.choice(X.shape[0], k, replace=False)]
    centers = np.array([X[3], X[0]])

    for i in range(max_iters):
        # Assign clusters based on L1 distance
        distances = manhattan_distances(X, centers)
        labels = np.argmin(distances, axis=1)

        # Compute new cluster centers
        new_centers = np.array([X[labels == j].mean(axis=0) for j in range(k)])

        # Check for convergence
        if np.all(np.abs(new_centers - centers) < tol):
            break

        centers = new_centers

    return labels, centers


# Sample data: Replace this with your own dataset
# X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
X = np.array([[0, -6], [4, 4], [0, 0], [-5, 2]])

# Define the number of clusters
k = 2

# Perform K-means with L1 norm
labels, centers = kmeans_l1(X, k)

# Print cluster centers and labels
print(f"centers:\n{centers}")
print(f"labels:\n{labels}")

# Plot the data points and cluster centers
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='viridis', s=100, alpha=0.7)
plt.scatter(centers[:, 0], centers[:, 1], s=300, c='red', marker='X', label='Centroids')
plt.title('K-means Clustering with L1 Norm')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()
