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


def fourier_series(t, coeffs, degree=5, noise=0.00):
    """Fourier series of input t.

    Args:
        t (float): scalar input
        coeffs (ndarray): coefficient tensor of dimension ()
        degree (int): maximum degree of Fourier series
        noise (flaot): standard deviation of Gaussian noise added to output
    """
    scaling = 0.5 * 2 * np.pi
    res = coeffs[0, 0] + coeffs[0, 1]
    for frequency in range(1, degree + 1):
        res += coeffs[frequency, 0] * np.cos(frequency * scaling * t) + coeffs[
            frequency, 1
        ] * np.sin(frequency * scaling * t)
    return res + np.random.normal(loc=0, scale=noise)


def generate_two_curves(n_samples, n_features, degree, offset, noise):
    """Data generation procedure for 'two curves'.

    Args:
        n_samples (int): number of samples to generate
        n_features (int): dimension of the data samples
        degree (int): maximum degree of Fourier series
        offset (float): distance between two curves
        noise (float): standard deviation of Gaussian noise added to curves
    """
    fourier_coeffs = np.random.uniform(size=(n_features, degree + 1, 2))
    fourier_coeffs = fourier_coeffs / np.linalg.norm(fourier_coeffs)

    # first manifold
    A = np.zeros(shape=(n_samples // 2, n_features))
    for s in range(n_samples // 2):
        # sample a point on the curve
        t = np.random.rand()
        # embed this point
        # every component is computed by another Fourier series
        for i in range(n_features):
            A[s, i] = fourier_series(t, fourier_coeffs[i], degree=degree, noise=noise)

    # second manifold: use same fourier series, plus offset
    B = np.zeros(shape=(n_samples // 2, n_features))
    for s in range(n_samples // 2):
        t = np.random.rand()
        for i in range(n_features):
            B[s, i] = fourier_series(t, fourier_coeffs[i], degree=degree, noise=noise)
    B = np.add(B, offset)

    X = np.r_[A, B]
    y = np.array([-1] * (n_samples // 2) + [1] * (n_samples // 2))

    s = StandardScaler()
    X = s.fit_transform(X)

    return X, y
