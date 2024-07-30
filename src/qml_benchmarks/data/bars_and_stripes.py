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


def generate_bars_and_stripes(n_samples, height, width, noise_std):
    """Data generation procedure for 'bars and stripes'.

    Args:
        n_samples (int): number of data samples to produce
        height (int): number of pixels for image height
        width (int): number of pixels for image width
        noise_std (float): standard deviation of Gaussian noise added to the pixels
    Returns:
        (array): data labels. -1 corresponds to a bar, +1 to a stripe.
    """
    X = np.ones([n_samples, 1, height, width]) * -1
    y = np.zeros([n_samples])

    for i in range(len(X)):
        if np.random.rand() > 0.5:
            rows = np.where(np.random.rand(height) > 0.5)[0]
            X[i, 0, rows, :] = 1.0
            y[i] = +1
        else:
            columns = np.where(np.random.rand(width) > 0.5)[0]
            X[i, 0, :, columns] = 1.0
            y[i] = -1
        X[i, 0] = X[i, 0] + np.random.normal(0, noise_std, size=X[i, 0].shape)

    return X, y
