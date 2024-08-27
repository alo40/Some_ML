import numpy as np

V = np.array([0, 0, 0, 0])  # value function
V_new = np.zeros_like(V)
gamma = 0.75  # discount factor
R = np.array(
    [[1, 1, 10,  0],   # reward for action UP
     [0, 1,  1, 10]]   # reward for action DOWN
)
steps = 100

print(f"V_0 = {V_new}")
for step in range(1, steps):
    # old = new
    V = V_new
    V_new = np.zeros(len(V))

    # action UP
    A_up = R[0, 0] + gamma * V[1]
    B_up = R[0, 1] + gamma * V[2]
    C_up = R[0, 2] + gamma * V[3]
    D_up = 0

    # action DOWN
    A_down = 0
    B_down = R[1, 1] + gamma * V[1]
    C_down = R[1, 2] + gamma * V[2]
    D_down = R[1, 3] + gamma * V[3]

    # updates
    V_new[0] = max(A_up, A_down)
    V_new[1] = max(B_up, B_down)
    V_new[2] = max(C_up, C_down)
    V_new[3] = max(D_up, D_down)

    print(f"V_{step} = {V_new}")
