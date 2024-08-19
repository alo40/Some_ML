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

# Initialization
# ------------------------------------
X = np.loadtxt("test_incomplete.txt")
# X = np.loadtxt("test_complete.txt")
K = 4
seed = 0
mixture, posterior = common.init(X, K, seed)

# Test
# ------------------------------------
posterior_naive, LL_naive = naive_em.estep(X, mixture)
posterior, LL = em.estep(X, mixture)
pass

# # EM update (test)
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