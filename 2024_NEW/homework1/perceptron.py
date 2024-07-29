import numpy as np


def calculate_linear_perceptron(x, y, labels, theta, T):
    # theta = np.array([-3, -3, 3])
    n = len(x)
    for t in range(T):
        for i, (x_i, y_i, label) in enumerate(zip(x, y, labels)):

            # Calculate average loss before theta update
            avg_loss = calculate_average_loss(x, y, labels, theta)

            if label * (x_i * theta[1] + y_i * theta[2] + theta[0]) <= 0:
                # print status
                print(f'T = {T}, i = {i}, \u03B8_0 = {theta[0]:.2f}, \u03B8 = [{theta[1]:.2f}, {theta[2]:.2f}], MISTAKE')

                # Theta update
                # theta[0] = theta[0] + label
                theta[1] = theta[1] + label * x_i
                theta[2] = theta[2] + label * y_i
            else:
                # print status
                print(f'T = {T}, i = {i}, \u03B8_0 = {theta[0]:.2f}, \u03B8 = [{theta[1]:.2f}, {theta[2]:.2f}]')
        print()
    return theta


def calculate_mini_linear_perceptron(x_i, y_i, label, theta):
    if label * (x_i * theta[1] + y_i * theta[2] + theta[0]) <= 0:
        # print status
        print(f'\u03B8_0 = {theta[0]:.2f}, \u03B8 = [{theta[1]:.2f}, {theta[2]:.2f}], MISTAKE')

        # Theta update
        # theta[0] = theta[0] + label
        theta[1] = theta[1] + label * x_i
        theta[2] = theta[2] + label * y_i

    else:
        # print status
        print(f'\u03B8_0 = {theta[0]:.2f}, \u03B8 = [{theta[1]:.2f}, {theta[2]:.2f}]')
    return theta


def calculate_average_loss(x, y, labels, theta):

    # Calculate Loss function
    n = len(x)
    Loss_sum = 0
    for (x_i, y_i, label) in zip(x, y, labels):
        z = label * (x_i * theta[1] + y_i * theta[2] + theta[0])
        if  z >= 1:
            Loss_sum += 0
        else:
            Loss_sum += 1 - z

    # Return average loss
    return Loss_sum / n


def calculate_regularization(theta, lambda_term=1):

    # Calculate L2 norm
    norm = np.sqrt(theta[1]**2 + theta[2]**2)

    # Return regularization value
    return lambda_term / 2 * norm
