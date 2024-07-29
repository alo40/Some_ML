import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from utils import *

# Create a figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create data grid
n = 50  # number of grid points
m = 10  # +- size of grid
X, Y = np.meshgrid(np.linspace(-m, m, n), np.linspace(-m, m, n))

# Select and create surface
function = bowl_function
# function = sine_function
Z_surface = function(X, Y)

# Create plane
global x0, y0
x0 = select_random_2D_element(X)
y0 = select_random_2D_element(Y)
z0 = function(x0, y0)
a, b, c = plane_equation_parameters(x0, y0, function)
Z_plane = a * X + b * Y + c

# Initialize surfaces
global surface, plane, point
surface = ax.plot_surface(X, Y, Z_surface, color='blue', alpha=0.2)
plane = ax.plot_surface(X, Y, Z_plane, color='red', alpha=0.2)
point = ax.scatter(x0, y0, z0, color='red', s=50)
ax.set_zlim(-2, 2)

# Initialize text
text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

def init():
    pass
    # global x0, y0
    # x0 = select_random_2D_element(X)
    # y0 = select_random_2D_element(Y)


def update(frame):
    global surface, plane, point
    ax.clear()  # Clear the previous plane
    plane.remove()
    surface.remove()
    point.remove()

    # gradient descend
    eta = 1.0  # step size
    global x0, y0
    x0, y0 = gradient_descend(x0, y0, function, eta)
    z0 = function(x0, y0)

    a, b, c = plane_equation_parameters(x0, y0, function)
    Z_plane = a * X + b * Y + c

    Z_surface = function(X, Y)

    surface = ax.plot_surface(X, Y, Z_surface, color='blue', alpha=0.2)
    plane = ax.plot_surface(X, Y, Z_plane, color='red', alpha=0.2)
    point = ax.scatter(x0, y0, z0, color='red', s=50)
    text.set_text(f'Frame: {frame}')

    # ax.set_xlim(-5, 5)
    # ax.set_ylim(-5, 5)
    # ax.set_zlim(0, 40)

    # testing
    print(f"frame = {frame}")
    print(f"x0 = {x0}, y0 = {y0}")
    print("")

    return surface, plane, point, text,


# Create the animation
ani = FuncAnimation(fig, update, frames=200, interval=100, repeat=True, init_func=init)

# # Save the animation as a GIF
# ani.save('animation_sine.gif', writer=PillowWriter(fps=20))

# Display the animation
plt.show()
