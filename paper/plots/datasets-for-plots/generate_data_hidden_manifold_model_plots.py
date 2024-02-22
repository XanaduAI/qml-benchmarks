"""Generate data used for plot in Figure 5d"""

from benchmarks.generate_hidden_manifold import generate_hidden_manifold_model
from sklearn.model_selection import train_test_split
import numpy as np

np.random.seed(3)

manifold_dimension = 3
n_features = 3
n_samples = 300

X, y = generate_hidden_manifold_model(n_samples, n_features, manifold_dimension)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

name_train = (
    f"hidden_manifold_model/hidden_manifold-3d-{manifold_dimension}manifold_train.csv"
)
data_train = np.c_[X_train, y_train]
np.savetxt(data_train, data_train, delimiter=",")

name_test = (
    f"hidden_manifold_model/hidden_manifold-3d-{manifold_dimension}manifold_test.csv"
)
data_test = np.c_[X_test, y_test]
np.savetxt(name_test, data_test, delimiter=",")
