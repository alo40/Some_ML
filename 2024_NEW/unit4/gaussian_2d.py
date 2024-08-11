import numpy as np
import matplotlib.pyplot as plt
from gaussian_function_dict import *


### DATA ###################################################################

# gaussian data
n = 5
x_ = np.linspace(-n, n, 1000)
y_ = np.linspace(-n, n, 1000)
X, Y = np.meshgrid(x_, y_)

# # exercise data
# cluster_1 = np.array([[-1.2, -0.8], [-1, -1.2], [-0.8, -1]])
# cluster_2 = np.array([[ 1.2,  0.8], [ 1,  1.2], [ 0.8,  1]])
# clusters = [cluster_1, cluster_2]
# x = compute_x(clusters)  # used only for exercise data
# colors = ['red', 'blue']

# random clustered points
size = 10
p_x = np.random.uniform(low=-n, high=n, size=size)  # Random x coordinates
p_y = np.random.uniform(low=-n, high=n, size=size)
x = np.column_stack((p_x, p_y))

### ALGO ###################################################################

# Compute gaussians parameters
K = 2
delta = compute_delta_j(x, K)
n = compute_n_j(delta, K)
mu = compute_mu_j(x, delta, n, K)
sigma = compute_sigma_j(x, delta, n, mu, K)
rho = 0  # Correlation coefficient (not used, but is needed to calculate Z)

# Create 2d gaussians
Z_1 = gaussian_2d(X, Y, mu[0, 0], mu[0, 1], sigma[0], sigma[0], rho)
Z_2 = gaussian_2d(X, Y, mu[1, 0], mu[1, 1], sigma[1], sigma[1], rho)

### PLOT ####################################################################

# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))
datas = plt.scatter(x[:, 0], x[:, 1], color='blue', s=100)
gaussian_1 = ax.contour(X, Y, Z_1, cmap='cool', alpha=1)
gaussian_2 = ax.contour(X, Y, Z_2, cmap='cool', alpha=1)
# plt.colorbar(gaussian_1, label='Probability Density')  # if no values, the scale is from 0 to 1 (not working!)

# axis range
x_range = (min(x_), max(x_))
y_range = (min(y_), max(y_))

# Set axis ticks every 1 unit
plt.xticks(np.arange(x_range[0], x_range[1], 1))
plt.yticks(np.arange(x_range[0], x_range[1], 1))

# set bold axis lines
plt.axhline(0, color='black', linewidth=1.5)
plt.axvline(0, color='black', linewidth=1.5)

# Set axis limits
ax = plt.gca()
ax.set_xlim([x_range[0], x_range[1]])
ax.set_ylim([y_range[0], y_range[1]])

# Set aspect ratio to be equal
plt.gca().set_aspect('equal', adjustable='box')

# Set labels and title
ax.set_title('2D Gaussian Distribution')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
plt.grid(linestyle='--')
plt.show()

