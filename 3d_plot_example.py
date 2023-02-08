# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 11:20:17 2023

@author: fori
"""

## libraries
# from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

## figure config
ax = plt.figure().add_subplot(projection='3d')

## plane def from general form 
## enter values without solving for z, e.g.
## 10x + 8y + 3z -155 = 0 -> a=10, b=8, c=3, d=-155
def plane_general_form(x, y, a, b, c, d):
    return (a*x + b*y + d)/(-c)

## data grids
x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)

## ------------------------------------------------------------
# ## example 1
# ## plane general form coefficients
# a = 10
# b = 8
# c = 3 
# d = -155

# ## point in plane coordinates
# p_x = 10
# p_y = 5
# p_z = 5

## ------------------------------------------------------------
## example 2 
# ## plane general form coefficients
# a = 1
# b = 4
# c = -2 
# d = -5

# ## arbitrary point in plane P_0
# x_0 = 1
# y_0 = -1
# z_0 = (a*x_0 + b*y_0 + d)/(-c)
# P_0 = np.array([x_0, y_0, z_0])

# ## arbitrary point in space P_1 (for this example is the same as the normal vector)
# x_1 = a
# y_1 = b
# z_1 = c
# P_1 = np.array([x_1, y_1, z_1])

## ------------------------------------------------------------
# ## example 3
# ## plane general form coefficients
# a = 5
# b = 3
# c = 1 
# d = -8

# ## point in plane coordinates
# p_x = 1
# p_y = 1
# p_z = 0

## ------------------------------------------------------------
## example 4
## plane general form coefficients
a = 3
b = 9
c = 1 
d = -8

## arbitrary point in plane P_0
x_0 = -1
y_0 = 1
z_0 = 2
P_0 = np.array([x_0, y_0, z_0])

## arbitrary point in plane P_1
x_1 = -4
y_1 = 2
z_1 = 2
P_1 = np.array([x_1, y_1, z_1])

## arbitrary point in plane P_2
x_2 = -2
y_2 = 1
z_2 = 5
P_2 = np.array([x_2, y_2, z_2])

## plot vectors from origin
ax.plot([x_0, 0], [y_0, 0], [z_0, 0], c='red', marker='.') # offset vector
ax.plot([x_1, 0], [y_1, 0], [z_1, 0], c='green', marker='.')
ax.plot([x_2, 0], [y_2, 0], [z_2, 0], c='green', marker='.')

## plot substraction vectors
v1 = P_1-P_0
ax.plot([v1[0]+P_0[0], 0+P_0[0]], 
        [v1[1]+P_0[1], 0+P_0[1]], 
        [v1[2]+P_0[2], 0+P_0[2]], c='blue', marker='.')
##
v2 = P_2-P_0
ax.plot([v2[0]+P_0[0], 0+P_0[0]], 
        [v2[1]+P_0[1], 0+P_0[1]], 
        [v2[2]+P_0[2], 0+P_0[2]], c='blue', marker='.')

# ## normal vector from origin
# n_0 = np.cross(P_1, P_2)
# ax.plot([n_0[0], 0], [n_0[1], 0], [n_0[2], 0], c='purple', marker='.')

# ## normal vector from origin using general form coeff
# ax.plot([a, 0], 
#         [b, 0], 
#         [c, 0], c='purple', marker='.')

## normal vector from plane using dot product
n = np.cross(v1, v2)
nu = n/np.linalg.norm(n) # normalized
ax.plot([nu[0]+P_0[0], 0+P_0[0]], 
        [nu[1]+P_0[1], 0+P_0[1]], 
        [nu[2]+P_0[2], 0+P_0[2]], c='purple', marker='.')

# ## plane
# Z = plane_general_form(X, Y, a, b, c, d)
# ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, alpha=0.1, rstride=8, cstride=8)

## ------------------------------------------------------------
## plotting

## origin
ax.scatter(0, 0, 0, c='red', marker='.', s=100)

## plane
# Z = plane_general_form(X, Y, a, b, c, d)
# ax.plot_surface(X, Y, Z, 
#                 edgecolor='royalblue', 
#                 lw=0.5, 
#                 rstride=8, 
#                 cstride=8,
#                 alpha=0.1)

## point(s) in plane
# ax.scatter(x_0, y_0, z_0, c='green', marker='.', s=100)
# ax.scatter(x_1, y_1, z_1, c='green', marker='.', s=100)
# ax.scatter(x_2, y_2, z_2, c='green', marker='.', s=100)

# ## plot vectors in plane
# ax.plot([x_1, x_0], [y_1, y_0], [z_1, z_0], c='green')
# ax.plot([x_2, x_0], [y_2, y_0], [z_2, z_0], c='green')
# ax.plot([n[0], x_0], [n[1], y_0], [n[2], z_0], c='purple')

## plot vectors from origin
# ax.plot([x_1, 0], [y_1, 0], [z_1, 0], c='green')
# ax.plot([x_2, 0], [y_2, 0], [z_2, 0], c='green')
# n = np.cross([x_1, y_1, z_1], [x_2, y_2, z_2])
# ax.plot([n[0], 0], [n[1], 0], [n[2], 0], c='purple')

# ## unit normal vector
# N = np.array([a, b, c])
# n = N/np.sqrt(a**2 + b**2 + c**2)

## plot vector P_1 to P0
# ax.plot([x_0, x_1], [y_0, y_1], [z_0, z_1], c='purple')

# ## closest point in plane to arbitrary point in space
# v = [a, b, c]
# norm_v = np.sqrt(a**2 + b**2 + c**2)

## plot normal vector (is not a POSITION vector)
# n_x = [0, a]
# n_y = [0, b]
# n_z = [0, c]
# ax.plot(n_x, n_y, n_z, c='purple')
# ax.scatter(a, b, c, c='purple', marker='.', s=100)

## ------------------------------------------------------------
## plot config
ax.set(xlabel='X', ylabel='Y', zlabel='Z')
ax.set(xlim=(-4, 4), ylim=(-4, 4), zlim=(-4, 4))
# ax.set_yticklabels([])
# ax.set_xticklabels([])
# ax.set_zticklabels([])
plt.show()

## not used
# t = np.linspace(-5, 5, 21)
# ax.plot(t+2, 4*t-4, -2*t+3, c='green')