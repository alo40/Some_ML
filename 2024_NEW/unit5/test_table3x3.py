import numpy as np
import matplotlib.pyplot as plt

# Generate the markov 3x3 grid
G = np.ones((3, 3)) * -10
G[1, 1] = 255  # invalid state
G[0, 0] = 0
G[2, 0] = -9  # goal
G[1, 0] = -11
G = np.flip(G, axis=1)
G = np.flip(G, axis=0)
# print(f"G_flipped = \n{np.flip(G, axis=1)}")  # flipping is only for visualization

# Create a figure and axis with a square aspect ratio
fig, ax = plt.subplots(figsize=(6, 6))  # Square figure size

# Display the grid as an image
ax.imshow(np.zeros_like(G), cmap='Pastel1', interpolation='none')

# Add text annotations for each cell
for i in range(G.shape[0]):
    for j in range(G.shape[1]):
        ax.text(j, i, f'{G[i, j]}', va='center', ha='center', fontsize=16)

# Set the ticks and labels to be empty
ax.set_xticks([])
ax.set_yticks([])

# Add grid lines for better visualization
ax.grid(which='both', color='black', linestyle='-', linewidth=2)

# Set aspect ratio to equal and ensure square grid
ax.set_aspect('equal')

# Set limits to ensure the grid is centered and has equal width and height
ax.set_xlim(-0.5, G.shape[1] - 0.5)
ax.set_ylim(G.shape[0] - 0.5, -0.5)

# Show the plot
plt.show()
