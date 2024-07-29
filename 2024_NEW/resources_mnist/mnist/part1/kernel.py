import numpy as np

### Functions for you to fill in ###



def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    return (X @ np.transpose(Y) + c)**p



def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    X_norm = np.sum(X ** 2, axis=-1)
    Y_norm = np.sum(Y ** 2, axis=-1)
    K = np.exp(-gamma * (X_norm[:, None] + Y_norm[None, :] - 2 * np.dot(X, Y.T)))
    return K

    #
    # norm = np.sqrt(X @ np.transpose(X) + Y @ np.transpose(Y))
    # return np.exp(-gamma * norm @ norm)
    #
    # return np.exp(gamma * (2 * X @ np.transpose(Y) - X @ np.transpose(X) - Y @ np.transpose(Y)))  # almost


X = np.array([0, 1, 2]).reshape(3, 1)
Y = np.array([0, 1, 2]).reshape(3, 1)
c = 1
p = 1
gamma = 1
kernel_poly = polynomial_kernel(X, Y, c, p)
kernel_rbf = rbf_kernel(X, Y, gamma)

# X = np.array([[0.47983256, 0.53993848, 0.89745637],
#  [0.82602168, 0.13524866, 0.00756249],
#  [0.90840187, 0.08424378, 0.8887979 ],
#  [0.48381389, 0.2222125 , 0.16997306],
#  [0.9218594 , 0.18952941, 0.63605647],
#  [0.28293074, 0.09918147, 0.88002323],
#  [0.34722505, 0.1062266 , 0.35266232],
#  [0.94257856, 0.28275328, 0.94795757],
#  [0.17457515, 0.94243631, 0.61153689],
#  [0.39561151, 0.23017046, 0.88678486],
#  [0.90700127, 0.44431222, 0.80161327],
#  [0.27369452, 0.2566151 , 0.67912964],
#  [0.94156638, 0.80476118, 0.78694593],
#  [0.36682413, 0.98327435, 0.96684715],
#  [0.20688207, 0.06156676, 0.96453676],
#  [0.52369949, 0.67118133, 0.29384719],
#  [0.38863448, 0.88300933, 0.29262601],
#  [0.11175519, 0.92594736, 0.55647958]])
#
# Y = np.array([[0.40360563, 0.24193281, 0.46198506],
#  [0.15171289, 0.1372596 , 0.9760375 ],
#  [0.70023499, 0.38173227, 0.00427181],
#  [0.97611828, 0.17574924, 0.85870407],
#  [0.34643243, 0.65045265, 0.93048759],
#  [0.75432037, 0.45227554, 0.32593295],
#  [0.37357844, 0.96930436, 0.1792233 ],
#  [0.58821496, 0.44309405, 0.01719763],
#  [0.97963246, 0.30078707, 0.02328156],
#  [0.08101083, 0.00822292, 0.79548351],
#  [0.03226935, 0.36943541, 0.86465818],
#  [0.4116972 , 0.77003581, 0.40081579],
#  [0.78027923, 0.37913463, 0.90449494],
#  [0.90598043, 0.65203557, 0.80711224],
#  [0.6804034 , 0.81873742, 0.23145088],
#  [0.32513744, 0.70242253, 0.57176034],
#  [0.97678833, 0.88647475, 0.31998084],
#  [0.38300089, 0.02864507, 0.66866155]])
