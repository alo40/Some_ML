import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the parameters for the Gaussian distribution
mu_x, mu_y = 0, 0  # Mean of the distribution
sigma_x, sigma_y = 1, 1  # Standard deviations

# Create a grid of (x, y) coordinates
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)

# Compute the 3D Gaussian function
z = (1/(2 * np.pi * sigma_x * sigma_y)) * \
    np.exp(-((x - mu_x)**2 / (2 * sigma_x**2) + (y - mu_y)**2 / (2 * sigma_y**2)))

# Create a figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(x, y, z, cmap='viridis')

# Set labels
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# Set title
ax.set_title('3D Gaussian Distribution')

# Show the plot
plt.show()
