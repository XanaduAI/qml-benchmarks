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

"""Generate datasets for the BARS & STRIPES benchmark."""

import os
import numpy as np
from qml_benchmarks.data import generate_bars_and_stripes

os.makedirs("paper/benchmarks/bars_and_stripes", exist_ok=True)

n_samples_train = 1000
n_samples_test = 200
noise_std = 0.5

for size in [4, 8, 16, 32]:

    np.random.seed(42)

    width = size
    height = size

    X_train, y_train = generate_bars_and_stripes(
        n_samples_train, height, width, noise_std
    )
    X_test, y_test = generate_bars_and_stripes(n_samples_test, height, width, noise_std)

    path_train = f"paper/benchmarks/bars_and_stripes/bars_and_stripes_{height}_x_{width}_{noise_std}noise_train.csv"
    data_train = np.c_[np.reshape(X_train, [n_samples_train, -1]), y_train]
    np.savetxt(path_train, data_train, delimiter=",")

    path_test = f"paper/benchmarks/bars_and_stripes/bars_and_stripes_{height}_x_{width}_{noise_std}noise_test.csv"
    data_test = np.c_[np.reshape(X_test, [n_samples_test, -1]), y_test]
    np.savetxt(path_test, data_test, delimiter=",")
