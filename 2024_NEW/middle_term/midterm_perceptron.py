import numpy as np
import itertools
from matplotlib import pyplot as plt


def linear_perceptron_2d(x1, x2, labels, theta, T):

    # Generate random theta
    theta = np.random.uniform(-10, 10, size=(3))
    print(f"theta = {theta}")

    # Generate a permutation of indices
    permutation = np.random.permutation(range(len(x1)))
    perm_x1 = x1[permutation]
    perm_x2 = x2[permutation]
    perm_labels = labels[permutation]
    print(f"permutations = {permutation}")

    # average loss
    loss_array = np.zeros(1)

    for t in range(T):
        print(f"T = {t}")
        for i, (x1_i, x2_i, label) in enumerate(zip(perm_x1, perm_x2, perm_labels)):

            if label * (x1_i * theta[1] + x2_i * theta[2] + theta[0]) <= 0:

                # Calculate average loss before theta update
                avg_loss = calculate_average_loss(x1, x2, labels, theta)


                print(f'i = {i}, ({x1_i:.2f}, {x2_i:.2f}) {label}, avg loss = {avg_loss:.2f}, \u03B8 = {theta[1:]}, \u03B8_0 = {theta[0]:.2f}, MISTAKE')
                # error_count[i] += 1

                # Theta update
                theta[0] += label
                theta[1] += label * x1_i
                theta[2] += label * x2_i

            else:
                avg_loss = calculate_average_loss(x1, x2, labels, theta)
                print(f'i = {i}, ({x1_i:.2f}, {x2_i:.2f}) {label}, avg loss = {avg_loss:.2f}, \u03B8 = {theta[1:]}, \u03B8_0 = {theta[0]:.2f}')

            # store avg_loss
            loss_array = np.append(loss_array, avg_loss)

    plt.title('average loss convergence')
    plt.plot(loss_array)
    plt.show()
    return theta


def permutation_linear_perceptron_2d(x1, x2, labels, theta, T, error_real):
    # define permutations
    perm_count = 0
    permutations = list(itertools.permutations(np.arange(len(x1))))
    permutations = [perm for perm in permutations if perm[0] == 6]  # all permutation starting with i=x

    for perm in permutations:
        error_count = np.zeros(len(x1))
        theta = np.array([0, 0, 0])
        for t in range(T):
            print(f'T = {t}')
            for i in perm:
                if labels[i] * (x1[i] * theta[1] + x2[i] * theta[2] + theta[0]) <= 0:
                    theta[0] += labels[i]
                    theta[1] += labels[i] * x1[i]
                    theta[2] += labels[i] * x2[i]
                    error_count[i] += 1

        # print(f"perm = {perm}, error count = {error_count}")

        if np.array_equal(error_real, error_count.astype(int)):
            print(f"success!")
            print(f"perm = {perm}, error delta = {error_real - error_count}")
            break

        perm_count += 1
        print(f"progress: {perm_count / len(permutations) * 100:.6f} %, error delta = {error_real - error_count}")

    return theta


def calculate_average_loss(x1, x2, labels, theta):
    n = len(x1)
    Loss_sum = 0
    for (x1_i, x2_i, label) in zip(x1, x2, labels):
        z = label * (x1_i * theta[1] + x2_i * theta[2] + theta[0])
        if  z >= 1:
            Loss_sum += 0
        else:
            Loss_sum += 1 - z

    # Return average loss
    return Loss_sum / n


def calculate_perceptron_ndim(X, Y, T):
    theta = np.zeros(X.shape[1])
    theta_0 = 0
    for t in range(T):
        for x, y in zip(X, Y):
            if y * (np.dot(theta, x) + theta_0) <= 0:
                theta += y * x
                theta_0 += y
    return theta, theta_0


def calculate_kernel_perceptron(x, y, T):

    # initialize theta, theta_0, loss
    theta = np.zeros_like(feature_map(x[0]))
    theta_0 = 0
    # loss_array = np.zeros(1)  # testing

    # kernel perceptron
    # loss_count = 0  # testing
    alpha = np.zeros(len(x))
    for t in range(T):
        for i, (xi, yi) in enumerate(zip(x, y)):
            sum_term = 0
            for j, (xj, yj) in enumerate(zip(x, y)):
                sum_term += alpha[j] * yj * quadratic_kernel(xj, xi)
            if yi * (sum_term + theta_0) <= 0:
                alpha[i] += 1
                theta_0 += yi
            #     loss_count += 1  # testing
            # loss_count += 0 # testing
            # loss_array = np.append(loss_array, loss_count)  # testing

        # loss_array = np.append(loss_array, alpha.sum())

    # # print loss (for testing only)
    # plt.title('average loss convergence')
    # plt.plot(loss_array)
    # plt.show()

    # update theta
    # alpha = np.array([1, 65, 11, 31, 72, 30, 0, 21, 4, 15])  # just for the exercise 2
    for j, (xj, yj) in enumerate(zip(x, y)):
        theta += alpha[j] * yj * feature_map(xj)

    return theta, theta_0, alpha


def quadratic_kernel(x,x_):
    return np.dot(x,x_) ** 2


def feature_map(x):
    return np.array([x[0]**2, np.sqrt(2) * x[0] * x[1], x[1]**2])


def check_kernel_alpha(x, y, T, alpha_desired):
    # define permutations
    perm_count = 0
    permutations = list(itertools.permutations(np.arange(len(x))))

    for perm in permutations:
        x_perm = x[list(perm)]
        y_perm = y[list(perm)]
        theta, theta_0, alpha = calculate_kernel_perceptron(x_perm, y_perm, T)
        if np.array_equal(alpha, alpha_desired):
            break
        perm_count += 1
        print(f"progress: {perm_count / len(permutations) * 100:.6f} %, alpha delta = {alpha_desired- alpha}")

    return theta, theta_0, alpha