import numpy as np
import matplotlib.pyplot as plt

def sine_function(x, y):
    """
    Function with multiple local minima
    :param x:
    :param y:
    :return:
    """
    return np.sin(np.sqrt(x**2 + y**2))


def bowl_function(x, y):
    """
    Define the bowl-shaped function
    :param x:
    :param y:
    :return:
    """
    return x**2 + y**2


def plane_equation_parameters(x, y, f):
    """
    Define the plane equation parameters
    :param x:
    :param y:
    :param f:
    :return:
    """
    z = f(x, y)
    xy = np.array([x, y])
    a, b = numerical_gradient_2d(f, xy, h=1e-5)
    c = z - a * x - b * y
    return a, b, c


def select_random_2D_element(array):

    # Get the shape of the array
    rows, cols = array.shape

    # Randomly select a row and column index
    random_row = np.random.randint(rows)
    random_col = np.random.randint(cols)

    return array[random_row, random_col]


def numerical_gradient_2d(f, xy, h):
    """
    Compute the numerical gradient of the function f at (x, y) using central difference.

    Parameters:
    f (function): The function for which to compute the gradient.
    xy (numpy array): The point [x, y] at which to compute the gradient.
    h (float): The step size (default is 1e-5).

    Returns:
    numpy array: The gradient of the function at (x, y).
    """
    grad = np.zeros_like(xy)

    # Compute partial derivative with respect to x
    xy_plus_h = xy.copy()
    xy_minus_h = xy.copy()

    xy_plus_h[0] += h
    xy_minus_h[0] -= h

    f_plus_h = f(xy_plus_h[0], xy_plus_h[1])
    f_minus_h = f(xy_minus_h[0], xy_minus_h[1])

    grad[0] = (f_plus_h - f_minus_h) / (2 * h)

    # Compute partial derivative with respect to y
    xy_plus_h = xy.copy()
    xy_minus_h = xy.copy()

    xy_plus_h[1] += h
    xy_minus_h[1] -= h

    f_plus_h = f(xy_plus_h[0], xy_plus_h[1])
    f_minus_h = f(xy_minus_h[0], xy_minus_h[1])

    grad[1] = (f_plus_h - f_minus_h) / (2 * h)

    return grad


def gradient_descend(x0, y0, f, eta):
    """
    Gradient descend function
    :param x0:
    :param y0:
    :param f:
    :param eta:
    :return:
    """
    xy = np.array([x0, y0])
    xy += -eta * numerical_gradient_2d(f, xy, h=1e-5)
    return xy[0], xy[1]

# def gradient_descend(a, b, f, xy, eta):
#
#     update = eta * numerical_gradient_2d(f, xy, h=1e-5)
#     a += update[0]
#     b += update[1]
#
#     x = xy[0]
#     y = xy[1]
#     z = f(x, y)
#     c = z - a * x - b * y
#
#     x0 = a / 2
#     y0 = b / 2
#
#     return a, b, c, x0, y0


# ### Test code ####################################################
#
# # Create a grid of x and y values
# x_values = np.linspace(-5, 5, 50)
# y_values = np.linspace(-5, 5, 50)
# X, Y = np.meshgrid(x_values, y_values)
#
# # Compute the function values over the grid
# Z_bowl = bowl_function(X, Y)
#
# # Compute the gradient at point Z[x,y]
# x0 = 3.0
# y0 = -3.0
# z0 = bowl_function(x0, y0)
# xy = np.array([x0, y0])
# a, b = numerical_gradient_2d(bowl_function, xy, h=1e-5)
# c = z0 - a * x0 - b * y0
#
# # Compute the corresponding z values for the plane
# Z_plane = a * X + b * Y + c
#
# # Plot the bowl-shaped function
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z_bowl, color='blue', alpha=0.4)
# ax.plot_surface(X, Y, Z_plane, color='red', alpha=0.4)
# ax.scatter(x0, y0, z0, color='red', s=50)
#
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Bowl-shaped Function: $f(x, y) = x^2 + y^2$')
#
# plt.show()
