# import matplotlib.pyplot as plt
import numpy as np
# from scipy.stats import describe

### 7.

# if __name__ == '__main__':
#     x = np.random.normal(2, 3, size=10**4)
#     # z = x + x
#     q = 2 * x
#     plt.hist(x, bins=100, alpha=0.1, color="red")
#     # plt.hist(z, bins=100, alpha=0.1)
#     plt.hist(q, bins=100, alpha=0.1)
#     plt.show()
#     print('x', describe(x))
#     # print('z', describe(z))
#     print('q', describe(q))

### 10.

# A = np.array([[1,2,3],[4,5,6],[1,2,1]])
# g = np.array([2,1,3])
# print(f"matrix product {np.matmul(A,g)}")
# print(f"element-wise multiplication {np.multiply(A,g)}")
# print(f"dot product (scalar) {np.dot(A,g)}")

### 12

# A = np.array([[1,-1],[1,0]])
# print(f"rank of A: {np.linalg.matrix_rank(A)}")

# A = np.array([[1,3],[2,6]])
# B = np.array([[1,2],[2,1]])
# C = np.array([[1,1,0],[0,1,1],[0,0,1]])
# D = np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]])
#
# print(f"rank of A: {np.linalg.matrix_rank(A)}")
# print(f"rank of B: {np.linalg.matrix_rank(B)}")
# print(f"rank of C: {np.linalg.matrix_rank(C)}")
# print(f"rank of D: {np.linalg.matrix_rank(D)}")

### 13

# A = np.array([[1,2,3],[4,5,6],[1,2,1]])
# print(f"Determinant of A: {np.linalg.det(A)}")
# print(f"Determinant of A.T: {np.linalg.det(A.T)}")

### 15

A = np.array([[3,0],[0.5,2]])
# w, v = np.linalg.eig(A)
# print(w)
# print(v)

# x = np.array([2,3])
# print(np.matmul(A,x.T))

print(np.linalg.det(A))