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

"""Generate datasets for the  LINEARLY SEPARABLE benchmark."""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from qml_benchmarks.data import generate_linearly_separable

np.random.seed(42)

os.makedirs("paper/benchmarks/linearly_separable", exist_ok=True)

n_samples = 300

for n_features in range(2, 21):
    margin = 0.02 * n_features

    X, y = generate_linearly_separable(n_samples, n_features, margin)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    name_train = f"paper/benchmarks/linearly_separable/linearly_separable_{n_features}d_train.csv"
    data_train = np.c_[X_train, y_train]
    np.savetxt(name_train, data_train, delimiter=",")

    name_test = f"paper/benchmarks/linearly_separable/linearly_separable_{n_features}d_test.csv"
    data_test = np.c_[X_test, y_test]
    np.savetxt(name_test, data_test, delimiter=",")
