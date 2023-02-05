# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 11:20:17 2023

@author: fori
"""

# libraries
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

# figure config
ax = plt.figure().add_subplot(projection='3d')

# plane def from general form 
def plane(x, y):
    return (10*x + 8*y - 155)/(-3)

# data
x = np.linspace(-6, 6, 100)
y = np.linspace(-6, 6, 100)
X, Y = np.meshgrid(x, y)
Z = plane(X, Y)

# plot 
ax.plot_surface(X, Y, Z, 
                edgecolor='royalblue', 
                lw=0.5, 
                rstride=8, 
                cstride=8,
                alpha=0.2)

## plot config
ax.set(xlabel='X', ylabel='Y', zlabel='Z')
# ax.set(xlim=(-2, 2), ylim=(-2, 2), zlim=(-2, 2))
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_zticklabels([])
plt.show()