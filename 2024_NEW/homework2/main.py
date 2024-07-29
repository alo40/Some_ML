import numpy as np
import scikit_learn as sl

def main():
    U = np.array([6, 0, 3, 6])
    V = np.array([4, 2, 1])
    X = np.outer(U, V)
    Y = np.array([[5, 0, 7], [0, 2, 0], [4, 0, 0], [0, 3, 6]])

    # # squared error term
    # squared_error = 0
    # for a in range(Y.shape[0]):
    #     for i in range(Y.shape[1]):
    #         # print(f"Y[{a}, {i}] = {Y[a, i]}")  # for test
    #         if Y[a, i] == 0:
    #             pass
    #         else:
    #             squared_error += (Y[a, i] - X[a, i])**2
    # squared_error /= 2
    # print(f"squared error term = {squared_error}")

    # # regularization term
    # L = 1  # lambda
    # reg_term = L / 2 * (np.sum(U**2) + (np.sum(V**2)))
    # print(f"regularization term = {reg_term}")

    # estimate U when V is fixed
    L = 1  # lambda
    for a in range(Y.shape[0]):
        if Y[a, 0] == 0: v0 = 0
        else: v0 = V[0]
        if Y[a, 1] == 0: v1 = 0
        else: v1 = V[1]
        if Y[a, 2] == 0: v2 = 0
        else: v2 = V[2]
        u_a = (v0*Y[a,0] + v1*Y[a,1] + v2*Y[a,2]) / (L + (v0**2 + v1**2 + v2**2))
        print(f"u_{a} = {u_a}")


if __name__ == '__main__':
    main()
