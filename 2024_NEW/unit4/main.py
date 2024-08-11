import numpy as np

# def euclidian_distance(x, y):
#     return np.linalg.norm([x, y])

K = 2
x1 = np.array([-1, 2])
x2 = np.array([-2, 1])
x3 = np.array([-1, 0])
x4 = np.array([ 2, 1])
x5 = np.array([ 3, 2])
z1 = np.array([-1, 1])
z2 = np.array([ 2 ,2])

C1 = (x1, x2, x3)
C2 = (x4, x5)
C = (C1, C2)
Z = (z1, z2)

sum_total = 0
for i, (z, c) in enumerate(zip(Z, C), start=1):
    sum = 0
    for x in c:
        sum += (x[0] - z[0]) ** 2 + (x[1] - z[1]) ** 2
    print(f"C{i} = {sum}")
    sum_total += sum
print(f"C total = {sum_total}")
