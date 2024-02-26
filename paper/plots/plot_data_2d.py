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

"""
Script to reproduce the 2d plots in Fig. 5 in the plots.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
sns.set(rc={"figure.figsize": (3, 3)})
sns.set(font_scale=1.3)
sns.set_style("white")

# Set the Seaborn color palette
palette = sns.color_palette("deep")

matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False

paths = [
    ("datasets-for-plots/linearly_separable/linearly_separable_2d", "LINEARLY_SEPARABLE-2d"),
    ("datasets-for-plots/mnist_pca/mnist_3-5_2d", "MNIST-2d"),
    (
        "datasets-for-plots/hyperplanes_parity/hyperplanes-2d-from2d-2n",
        "HYPERPLANES-2d-2n",
    ),
    (
        "datasets-for-plots/hyperplanes_parity/hyperplanes-2d-from2d-5n",
        "HYPERPLANES-2d-5n",
    ),
]

for path, out_name in paths:
    data_train = pd.read_csv(path + "_train.csv", header=None)
    X_train = data_train.iloc[:, :-1].values
    y_train = data_train.iloc[:, -1].values
    X_train_pos = X_train[y_train == 1]
    X_train_neg = X_train[y_train == -1]

    data_test = pd.read_csv(path + "_test.csv", header=None)
    X_test = data_test.iloc[:, :-1].values
    y_test = data_test.iloc[:, -1].values
    X_test_pos = X_test[y_test == 1]
    X_test_neg = X_test[y_test == -1]

    fig = plt.figure()
    ax = plt.gca()

    plt.scatter(X_train_pos[:, 0], X_train_pos[:, 1], marker=".", c=np.array(palette[0]).reshape(1,-1))
    plt.scatter(X_train_neg[:, 0], X_train_neg[:, 1], marker=".", c=np.array(palette[1]).reshape(1,-1))
    plt.scatter(X_test_pos[:, 0], X_test_pos[:, 1], marker="x", c=np.array(palette[0]).reshape(1,-1))
    plt.scatter(X_test_neg[:, 0], X_test_neg[:, 1], marker="x", c=np.array(palette[1]).reshape(1,-1))
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("$x_1$", fontsize=30)
    plt.ylabel("$x_2$", fontsize=30)

    plt.tight_layout()
    plt.savefig(f"figures/{out_name}" + "_plot.png")
    plt.show()
