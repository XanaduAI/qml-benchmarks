"""Generate data used for plot in Figure 5f"""

import numpy as np
from sklearn.model_selection import train_test_split
from benchmarks.generate_hyperplanes import generate_hyperplanes_parity

np.random.seed(1)

n_samples = 300
n_features = 2
n_hyperplanes = 5
dim_hyperplanes = 2

X, y = generate_hyperplanes_parity(
    n_samples, n_features, n_hyperplanes, dim_hyperplanes
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

name_train = f"hyperplanes_parity/hyperplanes-2d-from{dim_hyperplanes}d-{n_hyperplanes}n_train.csv"
data_train = np.c_[X_train, y_train]
np.savetxt(name_train, data_train, delimiter=",")

name_test = f"hyperplanes_parity/hyperplanes-2d-from{dim_hyperplanes}d-{n_hyperplanes}n_test.csv"
data_test = np.c_[X_test, y_test]
np.savetxt(name_test, data_test, delimiter=",")
