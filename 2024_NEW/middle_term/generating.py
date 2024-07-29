import numpy as np


def data_generating_random(x_range, y_range, n):

    # generate data and labels
    x = np.random.randint(x_range[0], x_range[1], n)
    y = np.random.randint(y_range[0], y_range[1], n)
    labels = np.random.choice([-1, 1], n)

    return x, y, labels


def data_generating_grouping(x_range, y_range, n):

    # label: 1
    x_1 = np.random.uniform(x_range[0], 6, n)
    y_1 = np.random.uniform(y_range[0], 6, n)
    label_1 = np.ones(x_1.shape)

    # label: -1
    x_2 = np.random.uniform(-6, x_range[1], n)
    y_2 = np.random.uniform(-6, y_range[1], n)
    label_2 = np.full(x_2.shape, -1)

    # concatenate arrays
    x = np.concatenate((x_1, x_2))
    y = np.concatenate((y_1, y_2))
    labels = np.concatenate((label_1, label_2))

    return x, y, labels
