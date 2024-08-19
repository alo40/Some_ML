import matplotlib.pyplot as plt

# Sample data for the first plot
x1 = [1, 2, 3, 4, 5]
y1 = [1, 4, 9, 16, 25]

# Sample data for the second plot
x2 = [1, 2, 3, 4, 5]
y2 = [2, 3, 5, 7, 11]

# Create a figure and a set of subplots
fig, ax = plt.subplots(1, 2)  # 2 rows, 1 column

# Plot the first graph
ax[0].plot(x1, y1, 'r-', label='Square Numbers')
ax[0].set_title('First Plot')
ax[0].legend()

# Plot the second graph
ax[1].plot(x2, y2, 'b--', label='Prime Numbers')
ax[1].set_title('Second Plot')
ax[1].legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()
