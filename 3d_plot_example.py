# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 11:20:17 2023

@author: fori
"""

## libraries
from mpl_toolkits.mplot3d import axes3d
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
x = np.linspace(-6, 6, 100)
y = np.linspace(-6, 6, 100)
X, Y = np.meshgrid(x, y)

## data from plane general form
Z = plane_general_form(X, Y, 10, 8, 3, -155)

## plot origin
ax.scatter(0, 0, 0, c='red', marker='.', s=100)

## plot normal vector
x_n = [0,1]
y_n = [0,1]
z_n = [0,1]
ax.plot(x_n, y_n, z_n)
ax.scatter(x_n[1], y_n[1], z_n[1], c='red', marker='v', s=100)

# ## plot plane 
# ax.plot_surface(X, Y, Z, 
#                 edgecolor='royalblue', 
#                 lw=0.5, 
#                 rstride=8, 
#                 cstride=8,
#                 alpha=0.2)

## plot config
ax.set(xlabel='X', ylabel='Y', zlabel='Z')
# ax.set(xlim=(-2, 2), ylim=(-2, 2), zlim=(-2, 2))
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_zticklabels([])
plt.show()