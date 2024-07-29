import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs


def my_hinge_loss_2(X, y, coef, intercept):
    # Split coordinates into x1 and x2
    x1, x2 = X[:, 0], X[:, 1]

    # Calculate Loss function
    n = len(x1)
    Loss_sum = 0
    for (x1_i, x2_i, label) in zip(x1, x2, y):
        z = label * (x1_i * coef[0] + x2_i * coef[1] + intercept)
        if z >= 1:
            Loss_sum += 0
        else:
            Loss_sum += 1 - z

    # Return average loss
    return Loss_sum / n


# Midterm exercise 1 dataset
y = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])
X = np.array([(0, 0), (2, 0), (3, 0), (0, 2), (2, 2), (5, 1), (5, 2), (2, 4), (4, 4), (5, 5)])

# # Midterm exercise 2 dataset
# y = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])
# X = np.array([(0, 0), (0, 2), (1, 1), (2, 0), (3, 3), (1, 4), (4, 1), (4, 4), (5, 2), (5, 5)])

# # Generate a synthetic dataset
# X, y = make_blobs(n_samples=100, centers=2, random_state=6)

# Train the SVM with a linear kernel
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, y)

# Get the coefficients and intercept of the decision boundary
w = clf.coef_[0]
b = clf.intercept_[0]
print(f"w: {w}, b: {b}")

# Hinge Loss from own function
loss = my_hinge_loss_2(X, y, w, b)
print(f"Hinge Loss myself = {loss}")

# Calculate the slope and intercept for the plot
slope = -w[0] / w[1]
intercept = -b / w[1]

# Plot the decision boundary
xx = np.linspace(min(X[:, 0]), max(X[:, 0]))
yy = slope * xx + intercept

# Plot the margin lines
margin = 1 / np.linalg.norm(w)
yy_down = yy - np.sqrt(1 + slope**2) * margin
yy_up = yy + np.sqrt(1 + slope**2) * margin

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter')
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
            facecolors='none', edgecolors='k', label='Support Vectors')

# Set aspect ratio to be equal
plt.gca().set_aspect('equal', adjustable='box')

plt.legend()
plt.show()
