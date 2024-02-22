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

"""Generate datasets for the MNIST benchmarks. Note that these can be large."""

import os
import torch
import numpy as np
# we import explicitly from data.mnist here because some dependencies of
# mnist generation are large and should not be imported by default
from qml_benchmarks.data.mnist import generate_mnist

# generate the MNIST PCA benchmark
np.random.seed(42)

os.makedirs("mnist_pca", exist_ok=True)

digitA = 3
digitB = 5

for n_features in range(2, 21):
    X_train, X_test, y_train, y_test = generate_mnist(
        digitA, digitB, preprocessing="pca", n_features=n_features
    )

    name_train = f"mnist_pca/mnist_{digitA}-{digitB}_{n_features}d_train.csv"
    data_train = np.c_[X_train, y_train]
    np.savetxt(name_train, data_train, delimiter=",")

    name_test = f"mnist_pca/mnist_{digitA}-{digitB}_{n_features}d_test.csv"
    data_test = np.c_[X_test, y_test]
    np.savetxt(name_test, data_test, delimiter=",")

# generate the MNIST PCA- benchmark
np.random.seed(42)

os.makedirs("mnist_pca-", exist_ok=True)

digitA = 3
digitB = 5

for n_features in range(2, 21):
    X_train, X_test, y_train, y_test = generate_mnist(
        digitA, digitB, preprocessing="pca-", n_features=n_features, n_samples=250
    )

    name_train = f"mnist_pca-/mnist_{digitA}-{digitB}_{n_features}d-250_train.csv"
    data_train = np.c_[X_train, y_train]
    np.savetxt(name_train, data_train, delimiter=",")

    name_test = f"mnist_pca-/mnist_{digitA}-{digitB}_{n_features}d-250_test.csv"
    data_test = np.c_[X_test, y_test]
    np.savetxt(name_test, data_test, delimiter=",")

# generate the MNIST CG benchmark
torch.manual_seed(42)

os.makedirs("mnist_cg", exist_ok=True)

digitA = 3
digitB = 5

for height in [4, 8, 16, 32]:
    X_train, X_test, y_train, y_test = generate_mnist(
        digitA, digitB, preprocessing="cg", height=height
    )

    name_train = f"mnist_cg/mnist_pixels_{digitA}-{digitB}_{height}x{height}_train.csv"
    data_train = np.c_[X_train, y_train]
    np.savetxt(name_train, data_train, delimiter=",")

    name_test = f"mnist_cg/mnist_pixels_{digitA}-{digitB}_{height}x{height}_test.csv"
    data_test = np.c_[X_test, y_test]
    np.savetxt(name_test, data_test, delimiter=",")
