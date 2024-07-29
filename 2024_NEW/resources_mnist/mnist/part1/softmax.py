import sys
import time
sys.path.append("..")
import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse


def augment_feature_vector(X):
    """
    Adds the x[i][0] = 1 feature for each data point x[i].

    Args:
        X - a NumPy matrix of n data points, each with d - 1 features

    Returns: X_augment, an (n, d) NumPy array with the added feature for each datapoint
    """
    column_of_ones = np.zeros([len(X), 1]) + 1
    return np.hstack((column_of_ones, X))


def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    # ## for loop
    # epsilon = 1e-10
    # C = np.dot(X, np.transpose(theta)) / temp_parameter
    # H = np.zeros([theta.shape[0], X.shape[0]])
    # for i in range(X.shape[0]):
    #     for j in range(theta.shape[0]):
    #         H[j, i] = np.exp(np.dot(theta[j], X[i]) / temp_parameter - np.max(C[i]))
    #         if H[j, i] < epsilon: H[j, i] = 0  # epsilon check
    #     H[:, i] = 1 / H[:, i].sum() * H[:, i]
    # return H

    ### matrix
    # n = X.shape[0]
    # k = theta.shape[0]
    C = np.dot(theta, np.transpose(X)) / temp_parameter
    C_max = np.max(C, axis=0)
    exp_parameter = np.exp(C - C_max)
    norm_parameter = np.sum(exp_parameter, axis=0)
    return exp_parameter / norm_parameter

    # ### sparce (not properly implemented)
    # X_sparce = sparse.coo_array(X)
    # theta_sparce = sparse.coo_array(theta)
    # C = (theta_sparce.dot(np.transpose(X_sparce)) / temp_parameter).toarray()
    # C_max = np.max(C, axis=0)
    # exp_parameter = np.exp(C - C_max)
    # norm_parameter = np.sum(exp_parameter, axis=0)
    # return exp_parameter / norm_parameter


def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns
        c - the cost value (scalar)
    """
    ### for loop
    n = X.shape[0]
    k = theta.shape[0]
    J_sum = 0
    H = compute_probabilities(X, theta, temp_parameter)
    for i in range(n):
        for j in range(k):
            if Y[i] == j:
                J_sum += np.log(H[j, i])
            else:
                J_sum += 0
    return -1 / n * J_sum + lambda_factor / 2 * (theta ** 2).sum()

    # ### matrix (not working)
    # n = X.shape[0]
    # k = theta.shape[0]
    # H = compute_probabilities(X, theta, temp_parameter)
    # sum_parameter = 0
    # for i in range(n):
    #     sum_parameter += np.sum(np.log(H[:, i]) * (np.arange(H.shape[0]) == Y[i]).astype(int))
    # return -sum_parameter / n + lambda_factor / 2 * (theta ** 2).sum()


def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    ### original
    # n = X.shape[0]
    # k = theta.shape[0]
    # H = compute_probabilities(X, theta, temp_parameter)
    # gradient = np.zeros_like(theta)
    # for m in range(k):
    #     sum_parameter = np.sum(X * ((Y == m).astype(int) - H[m]).reshape([n, 1]), axis=0)
    #     gradient[m] = -sum_parameter / (n * temp_parameter) + lambda_factor * theta[m]
    # theta_update = theta - alpha * gradient
    # return theta_update

    ### sparce
    n = X.shape[0]
    d = X.shape[1]
    k = theta.shape[0]
    H = compute_probabilities(X, theta, temp_parameter)
    gradient = np.zeros_like(theta)
    for m in range(k):
        sum_parameter = np.sum(X * ((Y == m).astype(int) - H[m]).reshape([n, 1]), axis=0)
        gradient[m] = -sum_parameter / (n * temp_parameter) + lambda_factor * theta[m]
    theta_update = theta - alpha * gradient
    return theta_update


def update_y(train_y, test_y):
    """
    Changes the old digit labels for the training and test set for the new (mod 3)
    labels.

    Args:
        train_y - (n, ) NumPy array containing the labels (a number between 0-9)
                 for each datapoint in the training set
        test_y - (n, ) NumPy array containing the labels (a number between 0-9)
                for each datapoint in the test set

    Returns:
        train_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                     for each datapoint in the training set
        test_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                    for each datapoint in the test set
    """
    train_y_mod3 = np.mod(train_y, 3)
    test_y_mod3 = np.mod(test_y, 3)
    return train_y_mod3, test_y_mod3


def compute_test_error_mod3(X, Y, theta, temp_parameter):
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
    assigned_labels = get_classification(X, theta, temp_parameter)
    assigned_labels = np.mod(assigned_labels, 3)
    test_error = 1 - np.mean(assigned_labels == Y)
    return test_error


def softmax_regression(X, Y, temp_parameter, alpha, lambda_factor, k, num_iterations):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset
    with theta initialized to the all-zeros array. Here, theta is a k by d NumPy array
    where row j represents the parameters of our model for label j for
    j = 0, 1, ..., k-1

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d-1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        temp_parameter - the temperature parameter of softmax function (scalar)
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        k - the number of labels (scalar)
        num_iterations - the number of iterations to run gradient descent (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
        cost_function_progression - a Python list containing the cost calculated at each step of gradient descent
    """
    X = augment_feature_vector(X)
    theta = np.zeros([k, X.shape[1]])
    cost_function_progression = []
    print(f"T={temp_parameter}")
    t0 = time.time()
    for i in range(num_iterations):
        cost_function_progression.append(compute_cost_function(X, Y, theta, lambda_factor, temp_parameter))
        theta = run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter)
        print(f"i = {i}, t = {time.time() - t0}")
    return theta, cost_function_progression


def get_classification(X, theta, temp_parameter):
    """
    Makes predictions by classifying a given dataset

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d - 1 features)
        theta - (k, d) NumPy array where row j represents the parameters of our model for
                label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        Y - (n, ) NumPy array, containing the predicted label (a number between 0-9) for
            each data point
    """
    X = augment_feature_vector(X)
    probabilities = compute_probabilities(X, theta, temp_parameter)
    return np.argmax(probabilities, axis = 0)


def plot_cost_function_over_time(cost_function_history):
    plt.plot(range(len(cost_function_history)), cost_function_history)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()


def compute_test_error(X, Y, theta, temp_parameter):
    error_count = 0.
    assigned_labels = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(assigned_labels == Y)
