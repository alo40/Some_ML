# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 22:01:04 2023

@author: fori
"""

import numpy as np

def randomization(n):
    return np.random.random([n,1])

def operations(h, w):
    A = np.random.random([h,w])
    B = np.random.random([h,w])
    s = A + B
    return A, B, s

def norm(A, B):
    s = np.linalg.norm(A + B)
    return s 

def neural_network(inputs, weights):
    return np.tanh(np.matmul(inputs.transpose(),weights))
    
def scalar_function(x, y):
    if x <= y:
        return x*y
    else:
        return x/y
    
def vector_function(x, y):
    scalar_vectorized = np.vectorize(scalar_function)
    return scalar_vectorized(x, y)
    
weights = np.array([[1],[2]])
inputs = np.array([[1],[2]])

x = np.array([[3.0], [5.0]])
y = np.array([[5], [3]])
# z = y + 3