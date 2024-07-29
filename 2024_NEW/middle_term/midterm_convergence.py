from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.exceptions import ConvergenceWarning
import warnings
import numpy as np


# # Load example data
# X, y = load_iris(return_X_y=True)

# Midterm exercise 2 dataset
y = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])
X = np.array([(0, 0), (0, 2), (1, 1), (2, 0), (3, 3), (1, 4), (4, 1), (4, 4), (5, 2), (5, 5)])

# Fit the SVM model
svc = SVC(max_iter=1000)

# Catch convergence warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    svc.fit(X, y)

# Check for convergence
if svc.n_iter_ is not None and svc.n_iter_[0] == svc.max_iter:
    print("The SVM did not converge.")
else:
    print("The SVM converged.")
