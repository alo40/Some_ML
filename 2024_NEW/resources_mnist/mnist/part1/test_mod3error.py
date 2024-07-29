import numpy as np
import softmax

"""
Returns the error of these new labels when the classifier predicts the digit. (mod 3)

Args:
    X - (n, d - 1) NumPy array (n datapoints each with d - 1 features)
    Y - (n, ) NumPy array containing the labels (a number from 0-2) for each
        data point
    theta - (k, d) NumPy array, where row j represents the parameters of our
            model for label j
    temp_parameter - the temperature parameter of softmax function (scalar)

Returns:
    test_error - the error rate of the classifier (scalar)
"""

X = np.array([[ 0.74860258,  0.50693842,  0.40423833],
              [ 0.42446398, -0.8318066 ,  1.49984034],
              [-0.52796478, -1.52231371,  0.17759967],
              [ 1.75876288, -1.03162331, -0.55898486],
              [ 2.09131703, -0.30806734, -2.04985459]])
Y = np.array([1, 0, 1, 2, 1])

d = X.shape[1] + 1
k = 3

# theta = np.array([1, 2, 3, 4])
theta = 0
temp_parameter = 1

assigned_labels = softmax.get_classification(X, theta, temp_parameter)
error = 1 - np.mean(assigned_labels == Y)
pass