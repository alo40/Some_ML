import numpy as np
from matplotlib import pyplot as plt
from other_perceptron import *


def plot_initializing(x_range, y_range):

    # Set up the figure
    plt.figure()
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Linear Perceptron')

    # Set axis ticks every 1 unit
    plt.xticks(np.arange(x_range[0], x_range[1], 1))
    plt.yticks(np.arange(x_range[0], x_range[1], 1))

    # Set axis limits
    ax = plt.gca()
    ax.set_xlim([x_range[0], x_range[1]])
    ax.set_ylim([y_range[0], y_range[1]])

    # Set aspect ratio to be equal
    plt.gca().set_aspect('equal', adjustable='box')

    plt.axhline(0, color='black', linewidth=1.5)
    plt.axvline(0, color='black', linewidth=1.5)


def plot_data_points(x: np.ndarray[np.int64], y: np.ndarray[np.int64], labels: np.ndarray[np.int64]):

    """
       Plots the array of x,y coordinates into points in a scatter plot.
       For a -1 label, a red point is plotted and a "-1" annotation is used
       For a 1 label, a blue point is plotted and a "-1" annotation is used

       Parameters:
       - x : Array of x-coordinates for the points
       - y : Array of y-coordinates for the points
       - labels : Array of labels for the points

       Returns:
       - none
    """

    # Create a list of colors based on labels
    colors = ['blue' if label == 1 else 'red' for label in labels]

    # Plot the points
    plt.scatter(x, y, color=colors, s=200)

    # Annotate the points
    for i, (x, y) in enumerate(zip(x, y)):
        plt.text(x, y, f'{labels[i]}', fontsize=10, ha='center', va='center', color='white')


def plot_decision_line(a: np.float64, b: np.float64, c: np.float64, x_range: tuple = (-10, 10)):

    """
        Plots the line defined by the equation ax + by + c = 0.

        Parameters:
        - a (float): Coefficient of x.
        - b (float): Coefficient of y.
        - c (float): Constant term.
        - x_range (tuple): Range of x values for plotting the line.
    """

    x_values = np.linspace(x_range[0], x_range[1], 400)

    if b != 0:
        y_values = -(a * x_values + c) / b
        plt.plot(x_values, y_values, c='m', label=f"decision line: a={a:.2f} b={b:.2f} c={c:.2f}")
    else:
        y_values = np.zeros(len(x_values))
        x_line = -c / a
        plt.axvline(x=x_line, c='m', label=f"decision line: a={a:.2f} b={b:.2f} c={c:.2f}")

    # Optional: Plot the random initial decision line
    plot_theta_norm([c, a, b], x_values, y_values)


def plot_objective_function(x, y, labels, n):

    # initialize the theta arrays
    theta0 = 0.0
    array_theta1 = np.linspace(-10, 10, 100)
    array_theta2 = np.linspace(-10, 10, 100)
    theta1, theta2 = np.meshgrid(array_theta1, array_theta2)

    # Create empty array for the objective function
    objective_function = np.empty_like(theta1)

    # Calculate and fill the objective function array
    for i in range(theta1.shape[0]):
        for j in range(theta1.shape[1]):
            theta = np.array([theta0, theta1[i][j], theta2[i][j]])
            avg_loss = calculate_average_loss(x, y, labels, theta, n)
            regularization = calculate_regularization(theta, lambda_term=1)
            objective_function[i][j] = avg_loss + regularization

    # Create the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_surface(theta1, theta2, objective_function, cmap='viridis')

    # Set labels
    ax.set_xlabel('theta_1 axis')
    ax.set_ylabel('theta_2 axis')
    ax.set_zlabel('objective function axis')

    # Set the title
    ax.set_title('objective function | theta_0 = 0')

    # plt.show()


def plot_theta_norm(theta, x_values, y_values):

    norm = np.sqrt(theta[0]**2 + theta[1]**2 + theta[2]**2)
    theta_x = theta[1] / norm
    theta_y = theta[2] / norm

    i_middle = len(x_values)//2
    dx = x_values[i_middle]
    dy = y_values[i_middle]

    # start = [0+dx, 0+dy]
    # end = [theta_x+dx, theta_y+dy]

    # plt.plot(start, end, color='blue', label='Normal Vector')
    # plt.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', color='blue', lw=2))

    plt.quiver(0+dx, 0+dy, theta[1], theta[2],
               angles='xy', scale_units='xy', scale=1, color='blue', label='normal vector')

