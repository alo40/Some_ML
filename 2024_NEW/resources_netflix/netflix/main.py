# This was done on August 2024
# This was a second test added
# This was a third test added

import numpy as np
import kmeans
import common
import naive_em
import em
import random
import matplotlib.pyplot as plt
from matplotlib import cm

# X = np.loadtxt("toy_data.txt")

# 2. --------------------------------------------------------------------

# K = [1, 2, 3, 4]
# seeds = [0, 1, 2, 3, 4]
#
# for k in K:
#     print()
#     for seed in seeds:
#         mixture, post = common.init(X, k, seed)
#         mixture, post, cost = kmeans.run(X, mixture, post)
#         print(f"K={k}, seed={seed},cost={cost}")
# common.plot(X, mixture, post, title="test")

# 3. --------------------------------------------------------------------

# K = 2
# seed = 0
# # seed = random.randint(1, 100)
# mixture, posterior = common.init(X, K, seed)
# mixture, posterior, joint_log_likelihood = naive_em.run(X, mixture, posterior)
# common.plot(X, mixture, posterior, title='EM converged')

# 4. --------------------------------------------------------------------

# K = [4]
# seeds = [0]
#
# for k in K:
#     print()
#     for seed in seeds:
#         # data initialization
#         mixture, posterior = common.init(X, k, seed)
#
#         # k-means algorithm
#         mixture_km, posterior_km, cost_km = kmeans.run(X, mixture, posterior)
#
#         #  em algorithm
#         mixture_em, posterior_em, cost_em = naive_em.run(X, mixture, posterior)
#
#         print(f"K={k}, seed={seed}, cost_km={cost_km}, cost_em={cost_em}")
#
# mixtures = [mixture_km, mixture_em]
# posteriors = [posterior_km, posterior_em]
# titles = ["k-means", "em"]
# common.plot_two_plots(X, mixtures, posteriors, titles)

# 5. --------------------------------------------------------------------

# Ks = [1, 2, 3, 4]
# seeds = [0, 1, 2, 3, 4]
# for K in Ks:
#     print()
#     for seed in seeds:
#         mixture, posterior = common.init(X, K, seed)
#         mixture, posterior, joint_log_likelihood = naive_em.run(X, mixture, posterior)
#         BIC = common.bic(X, mixture, joint_log_likelihood)
#         print(f"K={K}, seed={seed}, LL={joint_log_likelihood}, BIC={BIC}")

# 7. --------------------------------------------------------------------

# # ------------------------------------
# # Test 1 data
# # ------------------------------------
# X = np.array(
# [[0.85794562, 0.84725174],
#  [0.6235637 , 0.38438171],
#  [0.29753461, 0.05671298],
#  [0.        , 0.47766512],
#  [0.        , 0.        ],
#  [0.3927848 , 0.        ],
#  [0.        , 0.64817187],
#  [0.36824154, 0.        ],
#  [0.        , 0.87008726],
#  [0.47360805, 0.        ],
#  [0.        , 0.        ],
#  [0.        , 0.        ],
#  [0.53737323, 0.75861562],
#  [0.10590761, 0.        ],
#  [0.18633234, 0.        ]])
# K = 6
# Mu = np.array(
# [[0.6235637 , 0.38438171],
#  [0.3927848 , 0.        ],
#  [0.        , 0.        ],
#  [0.        , 0.87008726],
#  [0.36824154, 0.        ],
#  [0.10590761, 0.        ]])
# Var = np.array([0.16865269, 0.14023295, 0.1637321,  0.3077471, 0.13718238, 0.14220473])
# P = np.array([0.1680912, 0.15835331, 0.21384187, 0.14223565, 0.14295074, 0.17452722])
# mixture = common.GaussianMixture(Mu, Var, P)
# expected_posterior = np.array(
# [[0.65087662, 0.05857439, 0.02234959, 0.20258382, 0.0460844 , 0.01953118],
#  [0.36462427, 0.20175055, 0.09281546, 0.06127579, 0.17543624, 0.1040977 ],
#  [0.10995174, 0.22464491, 0.20513252, 0.02839796, 0.21019956, 0.22167331],
#  [0.27996042, 0.13156734, 0.18479023, 0.14012134, 0.11793063, 0.14563005],
#  [0.1680912 , 0.15835331, 0.21384187, 0.14223565, 0.14295074, 0.17452722],
#  [0.17188253, 0.2079498 , 0.16224482, 0.0981313 , 0.18938262, 0.17040893],
#  [0.33305679, 0.09456056, 0.14652199, 0.23671559, 0.08347925, 0.10566582],
#  [0.1634873 , 0.20447446, 0.16926051, 0.09967819, 0.18702813, 0.1760714 ],
#  [0.34047752, 0.04761128, 0.08765585, 0.42923507, 0.04092387, 0.0540964 ],
#  [0.20164864, 0.21756366, 0.14029582, 0.09378665, 0.19519249, 0.15151274],
#  [0.1680912 , 0.15835331, 0.21384187, 0.14223565, 0.14295074, 0.17452722],
#  [0.1680912 , 0.15835331, 0.21384187, 0.14223565, 0.14295074, 0.17452722],
#  [0.47521046, 0.09942182, 0.06885849, 0.20917529, 0.08508798, 0.06224596],
#  [0.09128906, 0.15565204, 0.25208102, 0.12427594, 0.14824935, 0.22845259],
#  [0.11018277, 0.17234878, 0.22552021, 0.1149792 , 0.16231678, 0.21465225]])
# expected_joint_LL = -8.829390
#
# # Test 1
# # ------------------------------------
# posterior, joint_log_likelihood = em.estep(X, mixture)
# print(f"posterior comparison: {[posterior == expected_posterior]}")
# print(f"likelihood comparison: {[joint_log_likelihood == expected_joint_LL]}")
# pass

# # ------------------------------------
# # Test 2 data
# # ------------------------------------
# X = np.loadtxt("test_incomplete.txt")
# # X = np.loadtxt("test_complete.txt")
# K = 4
# seed = 0
# mixture, posterior = common.init(X, K, seed)
#
# # Test 2
# # ------------------------------------
# # posterior, joint_log_likelihood = em.estep(X, mixture)
# # mixture = em.mstep(X, posterior, mixture, min_var = .25)
# mixture, posterior, joint_log_likelihood = em.run(X, mixture, posterior)

# 8. --------------------------------------------------------------------

# X = np.loadtxt("toy_data.txt")  # for testing only
X = np.loadtxt("test_incomplete.txt")  # for testing only
# X = np.loadtxt("netflix_incomplete.txt")
Ks = [2]
seeds = [0]
for K in Ks:
    print()
    for seed in seeds:
        mixture, posterior = common.init(X, K, seed)
        mixture, posterior, joint_log_likelihood = em.run(X, mixture, posterior)
        # mixture, posterior, joint_log_likelihood = naive_em.run(X, mixture, posterior)
        BIC = common.bic(X, mixture, joint_log_likelihood)
        print(f"K={K}, seed={seed}, LL={joint_log_likelihood}, BIC={BIC}")

X_predicted = em.fill_matrix(X, mixture)

# plot test
common.plot(X, mixture, posterior, title='Netflix Problem')

# NOT USED --------------------------------------------------------------

# # Test 2 (not used)
# # ------------------------------------
# posterior_naive, LL_naive = naive_em.estep(X, mixture)
# posterior, LL = em.estep(X, mixture)
# print(f"posterior comparison: {[posterior == posterior_naive]}")
# print(f"likelihood comparison: {[LL == LL_naive]}")

# # EM update (test, not used)
# # ------------------------------------
# pdf_likelihood = mixture.p
# for j in range(K):
#     for i, x_i in enumerate(X):
#         # non-zero terms
#         Cu = np.where(x_i != 0)[0]
#         # print(f"i={i}, Cu={Cu}, x_i[Cu] = {x_i[Cu]}")
#
#         # gaussian pdf
#         norm_term = 1 / np.sqrt(2 * np.pi * mixture.var[j])
#         expo_term = np.exp(-0.5 * (x_i[Cu] - mixture.mu[j][Cu]) ** 2 / mixture.var[j])
#         pdf_i = norm_term * expo_term
#
#         # posterior constructor
#         pdf_likelihood[j] *= np.prod(pdf_i)
# posterior = pdf_likelihood.sum()
# print(posterior)

# # Data import (example, not used)
# # ------------------------------------
# X_complete = np.loadtxt("test_complete.txt")
# X_incomplete = np.loadtxt("test_incomplete.txt")

# # Plot raw data (not used)
# # ------------------------------------
# # Conditioning
# n, d = X.shape
# x, y = np.meshgrid(range(d), range(n))
# x = x.flatten()
# y = y.flatten()
# z = np.zeros_like(x)  # Base (all bars start at z=0)
# height = X.flatten()  # Height of each bar (this will be mapped to the colormap)
#
# # Normalizing the heights to [0, 1] for the colormap
# norm = plt.Normalize(height.min(), height.max())
# colors = cm.hot(norm(height))
#
# # Plotting
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.bar3d(x, y, z, dx=0.8, dy=0.8, dz=height, color=colors, alpha=0.8)
# plt.show()
