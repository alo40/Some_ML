import numpy as np
from matplotlib import pyplot as plt
from plotting import *
from generating import *
from perceptron import *

if __name__ == "__main__":

    # Define the random coordinates and labels of the n points
    x_range = [-10, 10]
    y_range = [-10, 10]
    n = 50

    # Initialize plot
    plot_initializing(x_range, y_range)

    # Case 0: Data from the exercise
    # x = np.array([-1, 0])
    # y = np.array([0, 1])
    # labels = np.array([1, 1])

    # # Case 1: Integer and randomly distributed
    x, y, labels = data_generating_random(x_range, y_range, n)

    # # Case 2: Float and in two groups
    # x, y, labels = data_generating_grouping(x_range, y_range, n)

    # # Optional: Define the initial theta parameters
    # theta = np.array([0, 0, 0])  # Case 0: zero
    # theta = np.random.uniform(-10, 10, 3)  # Case 1: random

    # Calculate linear Perceptron update
    theta_ini = np.array([0, 0, 0])
    theta = calculate_linear_perceptron(x, y, labels, theta_ini, T=2)

    # Plot the random data points
    plot_data_points(x, y, labels)

    # Plot the random initial decision line
    plot_decision_line(a=theta[1], b=theta[2], c=theta[0])

    # # Optional: Plot objective function
    # plot_objective_function(x, y, labels, n)

    # # Optional: Calculate average loss and regularization
    # avg_loss = calculate_average_loss(x, y, labels, theta, n)
    # regularization = calculate_regularization(theta, lambda_term=1)

    # Show plot
    plt.grid()
    plt.legend()
    plt.show()