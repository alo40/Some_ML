import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Example data
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])
x, y = np.meshgrid(x, y)

# Flatten x and y to create pairs of coordinates
x = x.flatten()
y = y.flatten()

# Example height values for the bars
z = np.zeros_like(x)  # Base (all bars start at z=0)
height = x + y  # Height of each bar (this will be mapped to the colormap)

# Normalize the heights to [0, 1] for the colormap
norm = plt.Normalize(height.min(), height.max())
colors = cm.viridis(norm(height))

# Create the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create the 3D bar plot with colors based on the colormap
ax.bar3d(x, y, z, dx=0.8, dy=0.8, dz=height, color=colors, alpha=0.8)

# Set labels
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# Add a color bar to show the mapping from height to color
mappable = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
mappable.set_array(height)
plt.colorbar(mappable, shrink=0.5, aspect=5)

# Show the plot
plt.show()
