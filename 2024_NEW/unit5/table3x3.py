import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# Parameters for the binomial distribution
n = 10  # number of trials/steps
p = 0.5  # probability of success in each trial

# Generate the markov grid
G = np.ones((3, 3)) * -10
G[1, 1] = 255  # invalid state
G[0, 0] = 0
G[2, 0] = -9  # goal
G[1, 0] = -11
print(f"G_flipped = \n{np.flip(G, axis=1)}")  # flipping is only for visualization

# Other parameters
x = np.arange(0, n)
pmf = np.zeros(n)
pmf_sum = 0
R = -10
gamma = 0.5
for i in range(n):
    pmf_sum += R * gamma ** i
    pmf[i] = pmf_sum

# Plot the binomial distribution as a bar chart
plt.bar(x, pmf, color='skyblue')

# Set x-axis ticks to be separated by 1 unit
plt.xticks(np.arange(0, n + 1, 1), rotation=90)

# Set the number of y-axis ticks (example: 5 ticks)
plt.gca().yaxis.set_major_locator(MaxNLocator(30))

# Add titles and labels
plt.title(f'Utility function')
plt.xlabel('Steps')
plt.ylabel('Comulative Rewards')

# Show the plot
plt.grid(True, linestyle='--')
plt.show()
