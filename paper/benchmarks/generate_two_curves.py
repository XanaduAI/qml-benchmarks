# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate datasets for the TWO CURVES and TWO CURVES DIFF benchmarks."""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from qml_benchmarks.data import generate_two_curves

np.random.seed(3)

os.makedirs("two_curves_diff", exist_ok=True)

n_samples = 300
degree = 5
offset = 0.1
noise = 0.01

for n_features in range(2, 21):
    X, y = generate_two_curves(n_samples, n_features, degree, offset, noise)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    name_train = f"two_curves_diff/two_curves-5degree-0.1offset-{n_features}d_train.csv"
    data_train = np.c_[X_train, y_train]
    np.savetxt(name_train, data_train, delimiter=",")

    name_test = f"two_curves_diff/two_curves-5degree-0.1offset-{n_features}d_test.csv"
    data_test = np.c_[X_test, y_test]
    np.savetxt(name_test, data_test, delimiter=",")

# generate the TWO CURVES DIFF benchmark

os.makedirs("two_curves_diff", exist_ok=True)

n_samples = 300
n_features = 10
noise = 0.01

for degree in range(2, 21):
    offset = 1 / (2 * degree)

    X, y = generate_two_curves(n_samples, n_features, degree, offset, noise)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    name_train = f"two_curves_diff/two_curves-10d-{degree}degree_train.csv"
    data_train = np.c_[X_train, y_train]
    np.savetxt(name_train, data_train, delimiter=",")

    name_test = f"two_curves_diff/two_curves-10d-{degree}degree_test.csv"
    data_test = np.c_[X_test, y_test]
    np.savetxt(name_test, data_test, delimiter=",")
