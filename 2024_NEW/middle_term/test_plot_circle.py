import numpy as np
import matplotlib.pyplot as plt

# Parameters for the circle
h, k = 5, 5  # Center of the circle
r = 3        # Radius of the circle

# Create a grid of points
x = np.linspace(h - r, h + r, 400)
y = np.linspace(k - r, k + r, 400)
x, y = np.meshgrid(x, y)

# Circle equation (x - h)^2 + (y - k)^2 = r^2
equation = (x - h)**2 + (y - k)**2 - r**2

# Plot the circle
plt.contour(x, y, equation, levels=[0], colors='b')

# Set labels for axes
plt.xlabel('X')
plt.ylabel('Y')

# Set the aspect ratio of the plot to be equal
plt.gca().set_aspect('equal')

# Set title
plt.title('Circle Plot Using Equation')

# Show the plot
plt.show()
