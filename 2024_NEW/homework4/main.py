import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import numpy as np

# Sample data: Replace this with your own dataset
# X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
X = np.array([[0, -6], [4, 4], [0, 0], [-5, 2]])

# Define the number of clusters
k = 2

# Create a KMeans instance and fit it to the data
initial_kmeans = np.array([X[3], X[0]])
kmeans = KMeans(n_clusters=k, init=initial_kmeans, random_state=0).fit(X)
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# # Create a KMedoids instance and fit it to the data
# initial_medoids = np.array([X[3], X[0]])
# kmedoids = KMedoids(n_clusters=k, init=initial_medoids, random_state=0).fit(X)
# centers = kmedoids.cluster_centers_
# labels = kmedoids.labels_

# Print cluster centers and labels
print(f"centers:\n{centers}")
print(f"labels:\n{labels}")

# Plot the data points and cluster centers
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='viridis', s=200, alpha=1)
plt.scatter(centers[:, 0], centers[:, 1], s=100, c='red', marker='X', label='Centroids')

# bold axis
plt.axhline(0, color='black', linewidth=1.5)
plt.axvline(0, color='black', linewidth=1.5)

# Set axis ticks every 1 unit
lim = 10
plt.xticks(np.arange(-lim, lim, 1))
plt.yticks(np.arange(-lim, lim, 1))
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
plt.gca().set_aspect('equal', adjustable='box')
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()
