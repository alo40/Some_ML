# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 16:47:00 2023

@author: fori
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# a,b,c,d = 1,2,3,4
a = 1
b = 4
c = -2 
d = -5

x = np.linspace(-1,1,10)
y = np.linspace(-1,1,10)

X,Y = np.meshgrid(x,y)
Z = (d - a*X - b*Y) / c

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(X, Y, Z)
ax.set(xlabel='X', ylabel='Y', zlabel='Z')

## plot origin
ax.scatter(0, 0, 0, c='red', marker='.', s=100)
# ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))