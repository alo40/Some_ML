import numpy as np

### 4

# n = 2
# A = np.random.rand(n,1)
# B = np.random.rand(n,1)
# s = np.linalg.norm(A+B)
# # print(s)

### 5

# inputs = np.random.rand(n,1)
# weights = np.random.rand(n,1)
# nn = np.tanh(weights.transpose() @ inputs)
# pass

### 6

# def scalar_function(x, y):
#     """
#     Returns the f(x,y) defined in the problem statement.
#     """
#     if x < y:
#         return x * y
#     else:
#         return x / y
#
#
# def vector_function(x, y):
#     """
#     Make sure vector_function can deal with vector input x,y
#     """
#     try:
#         check_vector_lengths(x, y)
#         vfunc = np.vectorize(scalar_function)
#         return vfunc(x, y)
#     except ValueError as e:
#         print(e)
#
#
#
# def check_vector_lengths(v1, v2):
#     if len(v1) != len(v2):
#         raise ValueError("Vectors are not of the same length")
#
#
# n = 4
# x = np.random.rand(n)
# y = np.random.rand(n)
# print(vector_function(x, y))
# pass

### 8

def get_sum_metrics(predictions, metrics=[]):
    for i in range(3):
        metrics.append(lambda x: x + i)

    sum_metrics = 0
    for metric in metrics:
        sum_metrics += metric(predictions)

    return sum_metrics

predictions = 1
print(get_sum_metrics(predictions, metrics=[]))