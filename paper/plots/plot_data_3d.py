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
Script to reproduce the 3d plots in Fig. 5 in the plots.
"""
import matplotlib
import matplotlib.pyplot as plt
import os
import seaborn as sns

# Set the Seaborn color palette
palette = sns.color_palette("deep")

os.makedirs("figures", exist_ok=True)
from mpl_toolkits import mplot3d
import pandas as pd
import seaborn as sns

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

sns.set(rc={"figure.figsize": (3, 3)})
sns.set(font_scale=1.3)
sns.set_style("white")

paths = [
    (
        "../benchmarks/linearly_separable/linearly_separable_3d",
        "LINEARLY_SEPARABLE-3d",
        (0, 1, 2),
    ),
    ("../benchmarks/mnist_pca/mnist_3-5_3d", "MNIST-3d", (0, 1, 2)),
    (
        "../benchmarks/two_curves_diff/two_curves-10d-2degree",
        "TWO_CURVES-10d-2deg",
        (0, 8, 3),
    ),
    (
        "../benchmarks/two_curves_diff/two_curves-10d-10degree",
        "TWO_CURVES-10d-10deg",
        (0, 8, 3),
    ),
    (
        "../benchmarks/two_curves_diff/two_curves-10d-20degree",
        "TWO_CURVES-10d-20deg",
        (0, 8, 3),
    ),
    (
        "datasets-for-plots/hidden_manifold_model/hidden_manifold-3d-1manifold",
        "HMM-3d-1m",
        (0, 1, 2),
    ),
    (
        "datasets-for-plots/hidden_manifold_model/hidden_manifold-3d-2manifold",
        "HMM-3d-2m",
        (0, 1, 2),
    ),
    (
        "datasets-for-plots/hidden_manifold_model/hidden_manifold-3d-3manifold",
        "HMM-3d-3m",
        (0, 1, 2),
    ),
]

for path, out_name, dims in paths:
    dim1 = dims[0]
    dim2 = dims[1]
    dim3 = dims[2]

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
    ax = plt.axes(projection="3d")

    ax.scatter3D(
        X_train_pos[:, dim1],
        X_train_pos[:, dim2],
        X_train_pos[:, dim3],
        marker=".",
        c=palette[0],
        label="train 1",
    )
    ax.scatter3D(
        X_train_neg[:, dim1],
        X_train_neg[:, dim2],
        X_train_neg[:, dim3],
        marker=".",
        c=palette[1],
        label="train -1",
    )
    ax.scatter3D(
        X_test_pos[:, dim1],
        X_test_pos[:, dim2],
        X_test_pos[:, dim3],
        marker="x",
        c=palette[0],
        label="test 1",
    )
    ax.scatter3D(
        X_test_neg[:, dim1],
        X_test_neg[:, dim2],
        X_test_neg[:, dim3],
        marker="x",
        c=palette[1],
        label="test -1",
    )

    ax.set_xlabel(f"$x_{dim1 + 1}$", fontsize=25)
    ax.set_ylabel(f"$x_{dim2 + 1}$", fontsize=25)
    ax.set_zlabel(f"$x_{dim3 + 1}$", fontsize=25)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.xaxis.labelpad = -20
    ax.yaxis.labelpad = -20
    ax.zaxis.labelpad = -20
    ax.xaxis.get_label().set_backgroundcolor("w")
    ax.yaxis.get_label().set_backgroundcolor("w")
    ax.zaxis.get_label().set_backgroundcolor("w")

    plt.savefig(f"figures/{out_name}" + "_plot.png")

    plt.show()
