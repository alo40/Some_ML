'''
taken from:
https://maviccprp.github.io/a-support-vector-machine-in-just-a-few-lines-of-python-code/
'''

import numpy as np

def calculate_mini_svm_sgd(xi, yi, w):

    # w = np.zeros(len(X[0]))
    # eta = 1
    # epochs = 1

    # for epoch in range(0,epochs):
    #     for i, x in enumerate(X):
    #         if (Y[i]*np.dot(X[i], w)) < 1:
    #             w = w + eta * ((X[i] * Y[i]) + (-2 * (1/epoch) * w))
    #         else:
    #             w = w + eta * (-2 * (1/epoch) * w)

    eta = 1
    lambda_term = 1
    if (yi * np.dot(xi, w[-2:])) < 1:
        w[-2:] = w[-2:] + eta * (xi * yi - 2 * lambda_term * w[-2:])
    else:
        w[-2:] = w[-2:] + eta * (-2 * lambda_term * w[-2:])
    return w