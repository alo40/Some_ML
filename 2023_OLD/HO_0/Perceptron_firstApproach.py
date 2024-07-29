# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 13:21:25 2023

@author: fori
"""

import numpy as np
import matplotlib.pyplot as plt


# function to calculate line normal to characteristic vector and its projection on that line
def line(x1, a, b, c):
    x2 = (-a*x1 - c)/b
    x1_offset = (-c-b**2*c)/(a+b**2/a)
    x2_offset = -b*(-c-1/a*x1_offset)
    xx_offset = np.array([x1_offset, x2_offset])
    return x2, xx_offset


# declare figure
fig, ax = plt.subplots()

# training data
x_training = np.array([-2, -2])  # training point
y_training = -1  # label of training point, it can take values [-1, 1]
ax.plot(x_training[0], x_training[1], marker="o")
ax.text(x_training[0], x_training[1], "P1")

# characteristic vector
theta = np.array([0, 0])
theta_0 = 0

# perceptron algorithm
for n in range(2):
    value = y_training*(np.dot(x_training, theta) + theta_0)
    if value <= 0:
        print(str(n) + " True,  value = " + str(value) + ", theta = " + str(theta))
        theta = theta + y_training*x_training
        theta_0 = theta_0 + y_training
    else:
        print(str(n) + " False, value = " + str(value) + ", theta = " + str(theta))

# call function to calculate line normal to characteristic vector and its projection on that line
x1_line = np.linspace(-10, 10, 100)
x2_line, x_offset = line(x1_line, theta[0], theta[1], theta_0)

# plot characteristic vector
ax.plot([0 + x_offset[0], theta[0] + x_offset[0]], [0 + x_offset[1], theta[1] + x_offset[1]], color='red')

# plot line normal to characteristic vector
ax.plot(x1_line, x2_line, color='purple')

# configure figure
ax.set(xlabel='x_1', ylabel='x_2')
lim = 4
ax.set(xlim=(-lim, lim), ylim=(-lim, lim))
ax.grid(linestyle='--')
ax.set_aspect('equal', 'box')
plt.show()
