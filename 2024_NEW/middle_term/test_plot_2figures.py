import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Data for the plots
x = np.linspace(0, 10, 100)
y = np.sin(x)
z = np.cos(x)

# Create a new figure with two subplots
fig = plt.figure()

# Create the 3D plot in the first subplot
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(x, y, z, label='3D Line')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('3D Plot')
ax1.legend()

# Create the 2D plot in the second subplot
ax2 = fig.add_subplot(122)
ax2.plot(x, y, label='2D Line')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('2D Plot')
ax2.legend()

# Show the plots
plt.tight_layout()
plt.show()
