import numpy as np

V = np.array([0, 0, 0, 0, 0, 0])  # value function
V_new = np.zeros_like(V)
gamma = 0.75  # discount factor
size = len(V)
R = np.zeros((size, size))

# Fill Reward matrix
for i in range(size):
    for j in range(size):
        if i == 0 and j == 0:
            R[i, j] = 0
        elif i == j:
            R[i, j] = 1 / np.sqrt(i + 4)
        else:
            R[i, j] = abs(j - i) ** (1 / 3)

# Transitions matrices
# Action C
T_C = np.array(
    [[1,   0,   0,   0,   0,   0],
     [0, 0.3,   0, 0.7,   0,   0],
     [0,   0, 0.3,   0, 0.7,   0],
     [0,   0,   0, 0.3,   0, 0.7],
     [0,   0,   0,   0,   1,   0],
     [0,   0,   0,   0,   0,   1]]
)
# Action M
T_M = np.array(
    [[1, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0],
     [0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 1, 0]]
)

# Value function update
print(f"V_0 = {V}\n")
steps = 2
for step in range(1, steps):
    # old = new
    V = V_new
    V_new = np.zeros(size)

    # action C
    Q_C = np.zeros(size)
    Q_C[0] = T_C[0, 0] * (R[0, 0] + gamma * V[0])
    Q_C[1] = T_C[1, 1] * (R[1, 1] + gamma * V[1]) + T_C[1, 3] * (R[1, 3] + gamma * V[3])
    Q_C[2] = T_C[2, 2] * (R[2, 2] + gamma * V[2]) + T_C[2, 4] * (R[2, 4] + gamma * V[4])
    Q_C[3] = T_C[3, 3] * (R[3, 3] + gamma * V[3]) + T_C[3, 5] * (R[3, 5] + gamma * V[5])
    Q_C[4] = T_C[4, 4] * (R[4, 4] + gamma * V[4])
    Q_C[5] = T_C[5, 5] * (R[5, 5] + gamma * V[5])

    # action M
    Q_M = np.zeros(size)
    Q_M[0] = T_M[0, 0] * (R[0, 0] + gamma * V[0])
    Q_M[1] = T_M[1, 0] * (R[1, 0] + gamma * V[0])
    Q_M[2] = T_M[2, 1] * (R[2, 1] + gamma * V[1])
    Q_M[3] = T_M[3, 2] * (R[3, 2] + gamma * V[2])
    Q_M[4] = T_M[4, 3] * (R[4, 3] + gamma * V[3])
    Q_M[5] = T_M[5, 4] * (R[5, 4] + gamma * V[4])

    # update
    V_new = np.maximum(Q_C, Q_M)
    print(f"Q_C{step} = {Q_C}")
    print(f"Q_M{step} = {Q_M}")
    print(f"V_{step}  = {V_new}\n")
