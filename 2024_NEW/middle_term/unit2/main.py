import numpy as np


def hinge_loss(x, y, theta):
    n = x.shape[0]  # number of training examples
    loss_sum = 0
    z_vector = y - np.dot(x, theta)
    for z in z_vector:
        if z >= 1:
            loss_sum += 0
        else:
            loss_sum += 1 - z
    return loss_sum / n


def square_error_loss(x, y, theta):
    n = x.shape[0]  # number of training examples
    z = y - np.dot(x, theta)
    return (z**2).sum() / 2 / n


def main():
    x1 = np.array([ 1, 0, 1])
    x2 = np.array([ 1, 1, 1])
    x3 = np.array([ 1, 1,-1])
    x4 = np.array([-1, 1, 1])
    x = np.array([x1, x2, x3, x4])
    y = np.array([2, 2.7,-0.7, 2])
    theta = np.array([0, 1, 2])
    print(f"hinge loss = {hinge_loss(x, y, theta)}")
    print(f"square error loss = {square_error_loss(x, y, theta)}")


if __name__ == '__main__':
    main()