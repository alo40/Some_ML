# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import hinge_loss
import numpy as np


def my_hinge_loss(X, y, coef, intercept):
    # Split coordinates into x1 and x2
    x1, x2 = X[:, 0], X[:, 1]

    # Calculate Loss function
    n = len(x1)
    Loss_sum = 0
    for (x1_i, x2_i, label) in zip(x1, x2, y):
        z = label * (x1_i * coef[0][0] + x2_i * coef[0][1] + intercept)
        if z >= 1:
            Loss_sum += 0
        else:
            Loss_sum += 1 - z

    # Return average loss
    return Loss_sum / n


# # we create 40 separable points
# X, y = make_blobs(n_samples=40, centers=2, random_state=6)

# Midterm dataset
y = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])
X = np.array([(0, 0), (2, 0), (3, 0), (0, 2), (2, 2), (5, 1), (5, 2), (2, 4), (4, 4), (5, 5)])

# fit the model, don't regularize for illustration purposes
clf = svm.SVC(kernel="linear", C=1)
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# margin
print(f"margin = {1 / np.linalg.norm(clf.coef_)}" )

# Hinge Loss from sklearn
loss = hinge_loss(y, clf.decision_function(X))
print(f"Hinge Loss sklear = {loss}")

# Hinge Loss from own function
loss = my_hinge_loss(X, y, clf.coef_/2, clf.intercept_/2)
print(f"Hinge Loss myself = {loss}")

# plot the decision function
ax = plt.gca()
DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    plot_method="contour",
    colors="k",
    levels=[-1, 0, 1],
    alpha=0.5,
    linestyles=["--", "-", "--"],
    ax=ax,
)
# plot support vectors
ax.scatter(
    clf.support_vectors_[:, 0],
    clf.support_vectors_[:, 1],
    s=100,
    linewidth=1,
    facecolors="none",
    edgecolors="k",
)
plt.show()

