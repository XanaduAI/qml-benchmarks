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

"""Generate 8blobs dataset."""

import os
import numpy as np
from qml_benchmarks.data import generate_8blobs


if __name__ == "__main__":
    os.makedirs("spin_blobs", exist_ok=True)
    path_train = "spin_blobs/8blobs_train.csv"
    path_test = "spin_blobs/8blobs_test.csv"

    X, y = generate_8blobs(num_samples=5000)
    np.savetxt(path_train, X, delimiter=",")

    X, y = generate_8blobs(num_samples=1000)
    np.savetxt(path_test, X, delimiter=",")
