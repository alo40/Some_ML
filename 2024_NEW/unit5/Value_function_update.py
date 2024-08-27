import numpy as np


# Parameters
size = 5
Vk = np.zeros(size)  # Value functions
R = np.zeros(size)  # Reward function
R[-1] = 1
g = 0.5  # Discount factor gamma
steps = 100  # for iteration

# a1: Agent choose to stay
T_a1 = np.array(
    [[1/2, 1/2, 1/2, 1/2, 1/2],  # stay j=j
     [1/2, 1/4, 1/4, 1/4, 0.0],  # move right j=j+1
     [0.0, 1/4, 1/4, 1/4, 1/2]]  # move left j=j-1
)

# a2: Agent choose to move right
T_a2 = np.array(
    [[2/3, 2/3, 2/3, 2/3, 1/2],  # stay j=j
     [1/3, 1/3, 1/3, 1/3, 0.0],  # move right j=j+1
     [0.0, 0.0, 0.0, 0.0, 1/2]]  # move left j=j-1
)

# a3: Agent choose to move left
T_a3 = np.array(
    [[1/2, 2/3, 2/3, 2/3, 2/3],  # stay j=j
     [1/2, 0.0, 0.0, 0.0, 0.0],  # move right j=j+1
     [0.0, 1/3, 1/3, 1/3, 1/3]]  # move left j=j-1
)


# Matrix approach
# --------------------------------------------------------------------------------
Ts = [T_a1, T_a2, T_a3]
Q = np.zeros((len(Ts), size))  # Q-function to find the max of each state (column)
Q_a1 = np.zeros(size)
Q_a2 = np.zeros(size)
Q_a3 = np.zeros(size)
Vk_new = np.zeros(size)
print(f"step=0, Vk={Vk_new}")  # initialization
for step in range(1, steps):
    Vk = Vk_new
    Vk_new = np.zeros(size)

    # Value function update
    for i, T in enumerate(Ts):
        # Q-function for action a
        Q_a1       = T[0]       * (R       + g * Vk      )  # stay j=j
        Q_a2[ :-1] = T[1][ :-1] * (R[ :-1] + g * Vk[1:  ])  # move right j=j+1
        Q_a3[1:  ] = T[2][1:  ] * (R[1:  ] + g * Vk[ :-1])  # move left j=j-1
        Q[i] = Q_a1 + Q_a2 + Q_a3
    Vk_new = np.max(Q, axis=0)

print(f"Matrix  approach: step={steps}, Vk={Vk_new}")


# element by element approach
# --------------------------------------------------------------------------------
Vk_new = np.zeros(size)
# print(f"step=0, Vk={Vk_new}")  # initialization
for step in range(1, steps):
    Vk = Vk_new
    Vk_new = np.zeros(size)

    # action stay
    s0 =                              1 / 2 * (R[0] + g * Vk[0]) + 1 / 2 * (R[0] + g * Vk[1])
    s1 = 1 / 4 * (R[1] + g * Vk[0]) + 1 / 2 * (R[1] + g * Vk[1]) + 1 / 4 * (R[1] + g * Vk[2])
    s2 = 1 / 4 * (R[2] + g * Vk[1]) + 1 / 2 * (R[2] + g * Vk[2]) + 1 / 4 * (R[2] + g * Vk[3])
    s3 = 1 / 4 * (R[3] + g * Vk[3]) + 1 / 2 * (R[3] + g * Vk[3]) + 1 / 4 * (R[3] + g * Vk[4])
    s4 = 1 / 2 * (R[4] + g * Vk[3]) + 1 / 2 * (R[4] + g * Vk[4])
    # action right
    r0 =                              2 / 3 * (R[0] + g * Vk[0]) + 1 / 3 * (R[0] + g * Vk[1])
    r1 =                              2 / 3 * (R[1] + g * Vk[1]) + 1 / 3 * (R[1] + g * Vk[2])
    r2 =                              2 / 3 * (R[2] + g * Vk[2]) + 1 / 3 * (R[2] + g * Vk[3])
    r3 =                              2 / 3 * (R[3] + g * Vk[3]) + 1 / 3 * (R[3] + g * Vk[4])
    r4 = 1 / 2 * (R[4] + g * Vk[3]) + 1 / 2 * (R[4] + g * Vk[4])
    # action left
    l0 =                              1 / 2 * (R[0] + g * Vk[0]) + 1 / 2 * (R[0] + g * Vk[1])
    l1 = 1 / 3 * (R[1] + g * Vk[0]) + 2 / 3 * (R[1] + g * Vk[1])
    l2 = 1 / 3 * (R[2] + g * Vk[1]) + 2 / 3 * (R[2] + g * Vk[2])
    l3 = 1 / 3 * (R[3] + g * Vk[2]) + 2 / 3 * (R[3] + g * Vk[3])
    l4 = 1 / 3 * (R[4] + g * Vk[3]) + 2 / 3 * (R[4] + g * Vk[4])
    # Value function
    Vk_new[0] = max(s0, l0, r0)
    Vk_new[1] = max(s1, l1, r1)
    Vk_new[2] = max(s2, l2, r2)
    Vk_new[3] = max(s3, l3, r3)
    Vk_new[4] = max(s4, l4, r4)

print(f"Element approach: step={steps}, Vk={Vk_new}")
