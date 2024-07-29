import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from plotting import *
from generating import *
from perceptron import *
from gradient_descent import *


# Create an initialization function
def init():
    dot_marking.set_data([], [])
    line_decision.set_data([], [])
    line_normal.set_data([], [])
    annotation.set_text('')
    return dot_marking, line_decision, line_normal, annotation,


# Create an update function
def update(frame, theta):

    # Get index number
    cycle = frame // (2*n)  # cycle number
    if frame < 2*n:
        index = frame
    else:
        index = frame - (2 * n) * cycle

    # Calculate loss before theta update
    avg_loss = calculate_average_loss(x, y, labels, theta)

    # # Update theta using the perceptron algorithm
    # # print(f"t={cycle}, i={index}, \u03B8={theta}, average loss={avg_loss:.4f}")
    # # print(f'\u03B8$_0$ = {theta[0]:.2f}, \u03B8 = [{theta[1]:.2f}, {theta[2]:.2f}]')
    # theta_ = calculate_mini_linear_perceptron(x[index], y[index], labels[index], theta)

    # Update theta using the stochastic gradient descent
    bias = -1
    xi = np.array([x[index], y[index]])
    theta_ = calculate_mini_svm_sgd(xi, labels[index], theta)

    # Calculate decision line x-y components
    x_line = np.linspace(x_range[0], x_range[1], 400)
    if theta_[2] == 0:
        y_line = np.zeros(len(x_line))
    else:
        y_line = -(theta_[1] * x_line + theta_[0]) / theta_[2]

    # Get offset for normal vector  
    i_middle = len(x_line) // 2
    dx = x_line[i_middle]
    dy = y_line[i_middle]

    # Calculate normal vector x-y components
    x_normal = np.array([0 + dx, theta_[1] + dx])
    y_normal = np.array([0 + dy, theta_[2] + dy])

    # Update all lines, dots and annotations
    dot_marking.set_data([x[index]], [y[index]])
    line_decision.set_data(x_line, y_line)
    line_normal.set_data(x_normal, y_normal)
    annotation.set_text(f'Frame: {frame}, Index: {index}, Avg. Loss = {avg_loss:.2f}, \n'
                        f'\u03B8$_0$ = {theta_[0]:.2f}, \u03B8 = [{theta_[1]:.2f}, {theta_[2]:.2f}]')

    return dot_marking, line_decision, line_normal, annotation,


if __name__ == "__main__":

    # Define the random coordinates and labels of the n points
    x_range = [-10, 10]
    y_range = [-10, 10]
    n = 1  # half the total number of points

    # Set up the figure and axis
    fig, ax = plt.subplots()

    # Set figure parameters
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    plt.xticks(np.arange(x_range[0], x_range[1]+1, 1))
    plt.yticks(np.arange(x_range[0], x_range[1]+1, 1))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axhline(0, color='black', linewidth=1.5)
    plt.axvline(0, color='black', linewidth=1.5)

    # Case 0: Data from the exercise
    # x = np.array([-1, 0])
    # y = np.array([0, 1])
    # labels = np.array([1, 1])
    # theta = np.array([0, 0, 0])

    # # Case 1: Random generate data in two groups
    x, y, labels = data_generating_grouping(x_range, y_range, n)
    theta = np.array([0, 0, 0])

    # Plot data points
    plot_data_points(x, y, labels)

    # Initialize the line object and the annotation
    dot_marking, = ax.plot([], [], 'go', markersize=20)
    line_decision, = ax.plot([], [], lw=2, c='b')
    line_normal, = ax.plot([], [], lw=2, c='b')
    annotation = ax.annotate('', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12)

    # Set up the animation
    ani = FuncAnimation(fig, update,
                        frames=range(2*n), fargs=(theta,), init_func=init, blit=True, repeat=False, interval=500)

    # Display the animation
    plt.grid()
    plt.show()
