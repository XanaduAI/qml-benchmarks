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

import numpy as np
from sklearn.preprocessing import StandardScaler


def perceptron(x, w, b):
    """Transforms inputs according to a perceptron.
    Args:
        x (ndarray): input of shape (dim_hyperplanes,)
        w (ndarray): input-to-hidden weight matrix of shape (dim_hyperplanes,)
        b (float): bias
    """
    if np.dot(w, x) + b > 0:
        return 1
    return 0


def predict(x, weights, biases):
    """Implements the parity prediction logic.

    Args:
        x (ndarray): input of shape (dim_hyperplanes,)
        weights (ndarray): array of weight vectors defining
            the orientation of the hyperplanes
        biases (ndarray): array of biases defining
            the offset of the hyperplanes
    """
    preds = [perceptron(x, w, b) for w, b in zip(weights, biases)]
    n_ones = np.sum(preds)
    if n_ones % 2 == 0:
        return 1
    else:
        return -1


def generate_hyperplanes_parity(n_samples, n_features, n_hyperplanes, dim_hyperplanes):
    """Data generation procedure for 'hyperplanes and parity'.

    Args:
        n_samples (int): number of samples to generate
        n_features (int): dimension of the data samples
        n_hyperplanes (int): number of hyperplanes to use for prediction
        dim_hyperplanes (int): dimension of space in which
            hyperplanes are defined
    """

    # define hyperplanes
    weights = np.random.uniform(size=(n_hyperplanes, dim_hyperplanes))
    biases = np.random.uniform(size=(n_hyperplanes,))

    # hack: initially create more data than we need,
    # and then subselect to get balanced classes
    X = np.random.normal(size=(4 * n_samples, dim_hyperplanes))
    y = np.array([predict(x, weights, biases) for x in X])
    A = X[y == 1]
    B = X[y == -1]
    assert len(A) >= n_samples // 2
    assert len(B) >= n_samples // 2
    X = np.r_[A[: n_samples // 2], B[: n_samples // 2]]
    y = np.array([-1] * (n_samples // 2) + [1] * (n_samples // 2))

    # embed into feature space by a linear transform
    emb_W = np.random.uniform(size=(dim_hyperplanes, n_features))
    X = X @ emb_W

    s = StandardScaler()
    X = s.fit_transform(X)

    return X, y
