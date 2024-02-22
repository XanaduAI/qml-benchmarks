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

"""Generate datasets for the HYPERPLANES DIFF benchmark."""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from qml_benchmarks.data import generate_hyperplanes_parity

np.random.seed(1)

os.makedirs("hyperplanes_diff", exist_ok=True)

n_features = 10
dim_hyperplanes = 3
n_samples = 300

for n_hyperplanes in range(2, 21):

    X, y = generate_hyperplanes_parity(
        n_samples, n_features, n_hyperplanes, dim_hyperplanes
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    name_train = f"hyperplanes_diff/hyperplanes-10d-from{dim_hyperplanes}d-{n_hyperplanes}n_train.csv"
    data_train = np.c_[X_train, y_train]
    np.savetxt(name_train, data_train, delimiter=",")

    name_test = f"hyperplanes_diff/hyperplanes-10d-from{dim_hyperplanes}d-{n_hyperplanes}n_test.csv"
    data_test = np.c_[X_test, y_test]
    np.savetxt(name_test, data_test, delimiter=",")
