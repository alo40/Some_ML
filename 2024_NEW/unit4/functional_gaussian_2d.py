import numpy as np
import matplotlib.pyplot as plt

# Define a function to create the Gaussian distribution data
def gaussian_distribution(mu, sigma, size=1000):
    return np.random.normal(mu, sigma, size)

# Define a function to create the x values for plotting
def create_x_values(mu, sigma, num_points=100):
    x_min = mu - 3 * sigma
    x_max = mu + 3 * sigma
    return np.linspace(x_min, x_max, num_points)

# Define a function to calculate the Gaussian probability density function (PDF)
def gaussian_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * np.sqrt(2 * np.pi))
    exponent = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return coeff * exponent

# created by me
def gaussian_pdf_vector(x, mu, sigma, d):
    coeff = 1.0 / (2 * np.pi * sigma ** 2) ** (d / 2)
    exponent = np.exp(-1 / (2 * sigma ** 2) * norm(x, mu) ** 2)
    return coeff * exponent

# created by me
def norm(x, z):
    return np.sqrt((x[0] - z[0]) ** 2 + (x[1] - z[1]) ** 2)

# for exercise in U4
dimension = 2
x_vector = np.array([1/np.sqrt(np.pi), 2])
mu_vector = np.array([0, 2])
sigma = np.sqrt(1/(2*np.pi))

# 1st approach
likelihood_1 = np.log(gaussian_pdf_vector(x_vector, mu_vector, sigma, dimension))
print(f"likelihood 1 = {likelihood_1}")

# 2nd approach
likelihood_2 = -np.log(2 * np.pi * sigma ** 2) * dimension / 2 - 1 / (2 * sigma ** 2) * norm(x_vector, mu_vector) ** 2
print(f"likelihood 2 = {likelihood_2}")

# # Parameters for the Gaussian distribution
# mu = 0       # Mean
# sigma = 1    # Standard deviation
#
# # Generate data
# data = gaussian_distribution(mu, sigma)
#
# # Create x values
# x_values = create_x_values(mu, sigma)
#
# # Compute the Gaussian PDF
# pdf_values = gaussian_pdf(x_values, mu, sigma)
#
# # Plot the Gaussian distribution
# plt.figure(figsize=(8, 6))
# plt.hist(data, bins=30, density=True, alpha=0.6, color='g', edgecolor='black', label='Histogram of data')
# plt.plot(x_values, pdf_values, 'r-', label='Gaussian PDF')
# plt.title('Gaussian Distribution')
# plt.xlabel('Value')
# plt.ylabel('Probability Density')
# plt.legend()
# plt.grid(True)
# plt.show()
