# import time
# import numpy as np
# import scipy.sparse as sparse
#
# ITER = 100
# K = 10
# N = 10000
#
# def naive(indices, k):
# 		mat = [[1 if i == j else 0 for j in range(k)] for i in indices]
# 		return np.array(mat).T
#
#
# def with_sparse(indices, k):
# 		n = len(indices)
# 		M = sparse.coo_matrix(([1]*n, (Y, range(n))), shape=(k,n)).toarray()
# 		return M
#
#
# Y = np.random.randint(0, K, size=N)
#
# t0 = time.time()
# for i in range(ITER):
# 		naive(Y, K)
# print(time.time() - t0)
#
#
# t0 = time.time()
# for i in range(ITER):
# 		with_sparse(Y, K)
# print(time.time() - t0)

#############################################################################################

import numpy as np
from scipy.sparse import coo_matrix
import time

# Create two dense matrices
matrix1 = np.random.randint(0, 10, size=(1000, 1000))
matrix2 = np.random.randint(0, 10, size=(1000, 1000))

# print("Matrix 1:")
# print(matrix1)
# print("\nMatrix 2:")
# print(matrix2)

# Convert dense matrices to sparse COO matrices
sparse_matrix1 = coo_matrix(matrix1)
sparse_matrix2 = coo_matrix(matrix2)

# print("\nSparse Matrix 1:")
# print(sparse_matrix1)
# print("\nSparse Matrix 2:")
# print(sparse_matrix2)

# Performance comparison

# Dense matrix multiplication
start_time = time.time()
result_dense = np.dot(matrix1, matrix2)
end_time = time.time()
print("\nDense matrix multiplication took {:.2f} seconds".format(end_time - start_time))

# Sparse matrix multiplication
start_time = time.time()
result_sparse = sparse_matrix1.dot(sparse_matrix2)
end_time = time.time()
print("Sparse matrix multiplication took {:.2f} seconds".format(end_time - start_time))

# # If needed, convert the sparse result back to a dense format
# result_dense_from_sparse = result_sparse.toarray()
# print("\nResult of sparse matrix multiplication (converted to dense):")
# print(result_dense_from_sparse)
