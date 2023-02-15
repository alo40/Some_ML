# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 13:21:25 2023

@author: fori
"""

import numpy as np
import matplotlib.pyplot as plt

def line(x, a, b, c):
    Y = (-a*x - c)/b
    x_offset = (-c-b**2*c)/(a+b**2/a)
    y_offset = -b*(-c-1/a*x_offset)
    offset = np.array([x_offset, y_offset])
    return Y, offset 

x = np.array([3, -1])
y = -1

theta = np.array([0, 0])
theta_0 = 0

for n in range(2):
    value = y*(np.dot(x, theta) + theta_0)
    if value <= 0:
        print(str(n) + " True,  value = " + str(value) + ", theta = " + str(theta))
        theta = theta + y*x
        theta_0 = theta_0 + y
    else:
        print(str(n) + " False, value = " + str(value) + ", theta = " + str(theta)) 

X = np.linspace(-10, 10, 100)
offset = np.zeros(2)
Y, offset = line(X, theta[0], theta[1], theta_0)

fig, ax = plt.subplots()

ax.plot(x[0], x[1], marker="o")
ax.text(x[0], x[1], "P1")
ax.plot([0 + offset[0], theta[0] + offset[0]],[0 + offset[1], theta[1] + offset[1]])
ax.plot(X, Y)

ax.set(xlabel='x_1', ylabel='x_2')
lim = 4
ax.set(xlim=(-lim, lim), ylim=(-lim, lim))
ax.grid(linestyle = '--')
ax.set_aspect('equal', 'box')