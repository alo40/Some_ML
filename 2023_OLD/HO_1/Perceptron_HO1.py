import numpy as np
import matplotlib.pyplot as plt


# function to calculate line normal to characteristic vector and its projection on that line
def normal_line(a, b, c, x_lim, y_lim):
    if a == 0:
        x1_line = np.linspace(-x_lim, x_lim, 100)
        x2_line = np.zeros(x1_line.shape)
    elif b == 0:
        x2_line = np.linspace(-y_lim, y_lim, 100)
        x1_line = np.zeros(x2_line.shape)
    else:
        x1_line = np.linspace(-x_lim, x_lim, 100)
        x2_line = (-a*x1_line - c)/b
    return x1_line, x2_line
    # x1_projection = (-c-b**2*c)/(a+b**2/a)
    # x2_projection = -b*(-c-1/a*x1_projection)
    # x_projection = np.array([x1_projection, x2_projection])
    # return = x_projection


# declare figure
fig, ax = plt.subplots()

# configure figure
lim_x = 12
lim_y = 12
ax.set(xlabel='x_1', ylabel='x_2')
ax.set(xlim=(-lim_x, lim_x), ylim=(-lim_y, lim_y))
ax.grid(linestyle='--')
ax.set_aspect('equal', 'box')

# training points
x1 = np.array([-4., -2, -1, 2, 1])  # x-coordinate
x2 = np.array([2, 1, -1, 2, -2])  # y-coordinate
y = np.array([1., 1., -1, -1, -1])  # labels

# characteristic vector
theta = np.array([-4., 2.])
theta_0 = -5.

# plot points
for i in range(y.size):
    if y[i] > 0:
        ax.scatter(x1[i], x2[i], c='b', s=4)
    else:
        ax.scatter(x1[i], x2[i], c='r', s=4)

# perceptron
# loop_order = [0]
loop_order = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
miss_class = 0  # number of miss-classification in a run
for i in loop_order:
    point_i = np.array([x1[i], x2[i]])
    if y[i]*(np.dot(theta, point_i) + theta_0) <= 0:
        theta = theta + y[i]*point_i
        theta_0 = theta_0 + y[i]
        miss_class = miss_class + 1

# line normal to characteristic vector
x1_classifier, x2_classifier = normal_line(theta[0], theta[1], theta_0, lim_x, lim_y)
ax.plot(x1_classifier, x2_classifier, c='m', lw=0.4)
plt.text(0, 1, 'miss classifications = {}'.format(miss_class), fontsize=8)
plt.text(0, 0, 'theta = [{0},{1}], theta_0 = {2}'.format(theta[0], theta[1], theta_0), fontsize=8)

# end code
plt.show()
