# Baby project
# 29.07.2024

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from plotting import *
from generating import *
# from perceptron import *
from midterm_perceptron import *


def main():

    # Dataset ----------------------------------------------------------------------------------------

    # # Testing dataset
    # n = 5
    # x_range = [-10, 10]
    # y_range = [-10, 10]
    # x1, x2, labels = data_generating_grouping(x_range, y_range, n)
    # coordinates = np.column_stack((x1, x2))

    # # Midterm exercise 1 dataset
    # labels = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])
    # coordinates = np.array([(0, 0), (2, 0), (3, 0), (0, 2), (2, 2), (5, 1), (5, 2), (2, 4), (4, 4), (5, 5)])
    # x1, x2 = coordinates[:, 0], coordinates[:, 1]  # Split coordinates into x1 and x2
    # # error_real = np.array([1, 9, 10, 5, 9, 11, 0, 3, 1, 1])

    # Midterm exercise 2 dataset
    labels = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])
    coordinates = np.array([(0, 0), (0, 2), (1, 1), (2, 0), (3, 3), (1, 4), (4, 1), (4, 4), (5, 2), (5, 5)])
    # x1, x2 = coordinates[:, 0], coordinates[:, 1]  # Split coordinates into x1 and x2

    # apply feature vector
    phi = np.vstack([feature_map(xi) for xi in coordinates])

    # Perceptron ----------------------------------------------------------------------------------------

    # # Calculate linear Perceptron
    # # T = 200
    # theta_ini = np.array([0, 0, 0])
    # # # theta = permutation_linear_perceptron_2d(x1, x2, labels, theta_ini, T, error_real)  # for exercise 1
    # theta_ = linear_perceptron_2d(x1, x2, labels, theta_ini, T=20) # for exercise 2

    # # Calculate linear Perceptron using ndim
    # theta, theta_0 = calculate_perceptron_ndim(phi, labels, T=1000)
    # print(theta, theta_0)

    # Calculate kernel Perceptron
    theta, theta_0, alpha = calculate_kernel_perceptron(coordinates, labels, T=200)

    # # Calculate kernel Perceptron for a random permutation
    # perm = np.random.permutation(range(len(coordinates)))
    # perm_coordinates = coordinates[perm]
    # perm_labels = labels[perm]
    # theta, theta_0, alpha = calculate_kernel_perceptron(perm_coordinates, perm_labels, T=200)

    # # Calculate kernel Perceptron for all permutation
    # T = 200
    # alpha_desired = np.array([1, 65, 11, 31, 72, 30, 0, 21, 4, 15])
    # theta, theta_0, alpha = check_kernel_alpha(coordinates, labels, T, alpha_desired)
    # print(f"theta = {theta}, theta_0 = {theta_0}, alpha = {alpha}")

    # # Simple 2d plot --------------------------------------------------------------------------------------

    # # Initialize plot
    # x_range = [-10, 10]
    # y_range = [-10, 10]
    # plot_initializing(x_range, y_range)
    #
    # # Plot the final decision line
    # # plot_decision_line(a=theta[1], b=theta[2], c=theta[0])
    # plot_decision_line(theta[0], theta[1], theta_0)
    #
    # # Plot the random data points
    # plot_data_points(x1, x2, labels)
    #
    # # Show plot
    # plt.grid()
    # plt.legend()
    # plt.show()

    # Plotting ------------------------------------------------------------------------------------------

    fig = plt.figure()

    # Plot 3d feature map dataset------------------------------------------------
    ax1 = fig.add_subplot(121, projection='3d')

    # define meshgrid
    x_line = np.linspace(-10 + min(phi[:,0]), 10 + max(phi[:,0]), 1000)
    y_line = np.linspace(-10 + min(phi[:,1]), 10 + max(phi[:,1]), 1000)
    x_mesh, y_mesh = np.meshgrid(x_line, y_line)

    # feature map
    phi = np.vstack([feature_map(xi) for xi in coordinates])
    ax1.scatter3D(phi[:, 0], phi[:, 1], phi[:, 2], c=labels, cmap='cool')

    # plot 3d decision plane
    z_mesh = (-theta[0] * x_mesh - theta[1] * y_mesh - theta_0) / theta[2]
    ax1.plot_surface(x_mesh, y_mesh, z_mesh, alpha=0.2, rstride=100, cstride=100)

    # Set labels for axes and title
    ax1.set_title('3D feature map dataset')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Set the limits
    ax1.set_xlim3d(min(phi[:,0]), max(phi[:,0]))
    ax1.set_ylim3d(min(phi[:,1]), max(phi[:,1]))
    ax1.set_zlim3d(min(phi[:,2]), max(phi[:,2]))

    # Plot 2d original dataset------------------------------------------------
    ax2 = fig.add_subplot(122)

    ax2.scatter(coordinates[:, 0], coordinates[:, 1], c=labels, cmap='cool')

    # plot 2d decision boundary
    # working!
    # contour = theta[0] * x_mesh ** 2 + theta[1] * np.sqrt(2) * x_mesh * y_mesh + theta[2] * y_mesh ** 2 + theta_0
    #
    # phi = feature_map([x_mesh, y_mesh])
    # contour = theta[0] * phi[0] + theta[1] * phi[1] + theta[2] * phi[2] + theta_0  # working!
    #
    phi_mesh = feature_map([x_mesh, y_mesh])
    contour = np.tensordot(theta, phi_mesh, axes=(0, 0)) + theta_0
    ax2.contour(x_line, y_line, contour, levels=[0])  # working!

    # Set labels for axes
    ax2.set_title('2D original dataset')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')

    # Set the limits
    ax2.set_xlim(min(coordinates[:,0]), max(coordinates[:,0]))
    ax2.set_ylim(min(coordinates[:,1]), max(coordinates[:,1]))

    # Show the plots
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

