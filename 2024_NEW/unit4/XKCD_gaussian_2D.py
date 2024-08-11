import numpy as np
import matplotlib.pyplot as plt

# Define the parameters for the Gaussian distribution
mu_x, mu_y = 0, 0  # Mean of the distribution
sigma_x, sigma_y = 1, 1  # Standard deviations
rho = 0  # Correlation coefficient

# Create a grid of (x, y) coordinates
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Compute the 2D Gaussian function
def gaussian(X, Y, mu_x, mu_y, sigma_x, sigma_y, rho):
    coeff = 1 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho**2))
    exponent = - (1 / (2 * (1 - rho**2))) * (
        ((X - mu_x)**2 / sigma_x**2) +
        ((Y - mu_y)**2 / sigma_y**2) -
        (2 * rho * (X - mu_x) * (Y - mu_y)) / (sigma_x * sigma_y)
    )
    return coeff * np.exp(exponent)

Z = gaussian(X, Y, mu_x, mu_y, sigma_x, sigma_y, rho)

# Use XKCD-style plotting
with plt.xkcd():
    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, cmap='hot', alpha=0.5)
    plt.colorbar(label='Probability Density')
    plt.title('2D Gaussian Distribution (XKCD Style)')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.gca().set_aspect('equal', adjustable='box')  # Set 1:1 aspect ratio
    plt.show()
