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
Script to reproduce Fig. 16 in the plots, showing the shapes of different kernels in 2d.
"""
import csv
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
from qml_benchmarks import models
from qml_benchmarks.hyperparam_search_utils import csv_to_dict

os.makedirs("figures", exist_ok=True)

sns.set(rc={"figure.figsize": (11.5, 4)})
sns.set(font_scale=1.3)
sns.set_style("white")

cmap = sns.diverging_palette(30, 255, l=60, as_cmap=True)


def linear_kernel(X1, X2):
    K = np.zeros(shape=(len(X1), len(X2)))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            K[i, j] = np.dot(x1, x2)
    return K


def rbf_kernel(X1, X2, gamma):
    K = np.zeros(shape=(len(X1), len(X2)))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            K[i, j] = np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)
    return K


X = np.arange(0, np.pi, 0.15)
Y = np.arange(0, np.pi, 0.15)
grid_flat = np.array([[x, y] for x in X for y in Y])
X_mesh, Y_mesh = np.meshgrid(X, Y)
zero_point = np.array([[np.pi / 2, np.pi / 2]])

datasets = {
    "mnist_pca": ["../results/mnist_pca-", "mnist_3-5_", "d-250", "MNIST PCA-"],
    "linearly-separable": [
        "../results/linearly_separable",
        "linearly_separable_",
        "d",
        "LINEARLY SEPARABLE",
    ],
    "hmm": [
        "../results/hidden_manifold",
        "hidden_manifold-6manifold-",
        "d",
        "HIDDEN MANIFOLD",
    ],
    "two-curves": [
        "../results/two_curves",
        "two_curves-5degree-0.1offset-",
        "d",
        "TWO CURVES",
    ],
}

for dataname, (
        path_to_datafolder,
        dataset_name0,
        dataset_name1,
        displayname,
) in datasets.items():
    dataset_name = dataset_name0 + str(2) + dataset_name1

    fig = plt.figure()

    # ----------------------

    model_name = "SVC"
    hyperparams_path = (
            path_to_datafolder
            + f"/{model_name}/"
            + model_name
            + "_"
            + dataset_name
            + "_GridSearchCV-best-hyperparams.csv"
    )
    best_hyperparams = csv_to_dict(hyperparams_path)
    gamma = best_hyperparams["gamma"]
    Z_flat = np.squeeze(rbf_kernel(zero_point, grid_flat, gamma))
    Z_flat = (Z_flat - Z_flat.min()) / (Z_flat.max() - Z_flat.min())
    Z = Z_flat.reshape((len(X), len(Y)))

    ax = fig.add_subplot(1, 5, 1, projection="3d")
    ax.set_zorder(5)
    surf = ax.plot_surface(
        X_mesh, Y_mesh, Z, cmap=cmap, linewidth=0, antialiased=False
    )
    ax.set_title(f"Support\n Vector\n Classifier")

    # ----------------------

    model_name = "SeparableKernelClassifier"
    hyperparams_path = (
            path_to_datafolder
            + f"/{model_name}/"
            + model_name
            + "_"
            + dataset_name
            + "_GridSearchCV-best-hyperparams.csv"
    )
    best_hyperparams = csv_to_dict(hyperparams_path)
    clf = models.SeparableKernelClassifier(**best_hyperparams)
    clf.initialize(n_features=2)
    grid_flat_trans = clf.transform(grid_flat)
    zero_point_trans = clf.transform(zero_point)
    Z_flat = np.squeeze(clf.precompute_kernel(zero_point_trans, grid_flat_trans))
    Z_flat = (Z_flat - Z_flat.min()) / (Z_flat.max() - Z_flat.min())
    Z = Z_flat.reshape((len(X), len(Y)))

    ax = fig.add_subplot(1, 5, 2, projection="3d")
    ax.set_zorder(4)
    surf = ax.plot_surface(
        X_mesh, Y_mesh, Z, cmap=cmap, linewidth=0, antialiased=False
    )
    ax.set_title(f"Separable\n Kernel\n Classifier")

    # ----------------------

    model_name = "IQPKernelClassifier"
    hyperparams_path = (
            path_to_datafolder
            + f"/{model_name}/"
            + model_name
            + "_"
            + dataset_name
            + "_GridSearchCV-best-hyperparams.csv"
    )
    best_hyperparams = csv_to_dict(hyperparams_path)
    clf = models.IQPKernelClassifier(**best_hyperparams)
    clf.initialize(n_features=2)
    grid_flat_trans = clf.transform(grid_flat)
    zero_point_trans = clf.transform(zero_point)
    Z_flat = np.squeeze(clf.precompute_kernel(zero_point_trans, grid_flat_trans))
    Z_flat = (Z_flat - Z_flat.min()) / (Z_flat.max() - Z_flat.min())
    Z = Z_flat.reshape((len(X), len(Y)))

    ax = fig.add_subplot(1, 5, 3, projection="3d")
    ax.set_zorder(3)
    surf = ax.plot_surface(
        X_mesh, Y_mesh, Z, cmap=cmap, linewidth=0, antialiased=False
    )
    ax.set_title(f"IQP\n Kernel\n Classifier")

    # ----------------------

    model_name = "QuantumKitchenSinks"
    hyperparams_path = (
            path_to_datafolder
            + f"/{model_name}/"
            + model_name
            + "_"
            + dataset_name
            + "_GridSearchCV-best-hyperparams.csv"
    )
    best_hyperparams = csv_to_dict(hyperparams_path)
    clf = models.QuantumKitchenSinks(**best_hyperparams)
    clf.initialize(n_features=2)
    grid_flat_trans = clf.transform(grid_flat)
    zero_point_trans = clf.transform(zero_point)
    Z_flat = np.squeeze(linear_kernel(zero_point_trans, grid_flat_trans))
    Z_flat = (Z_flat - Z_flat.min()) / (Z_flat.max() - Z_flat.min())
    Z = Z_flat.reshape((len(X), len(Y)))

    ax = fig.add_subplot(1, 5, 4, projection="3d")
    ax.set_zorder(1)
    surf = ax.plot_surface(
        X_mesh, Y_mesh, Z, cmap=cmap, linewidth=0, antialiased=False
    )
    ax.set_title(f"Quantum\n Kitchen\n Sinks")

    # ----------------------

    model_name = "ProjectedQuantumKernel"
    hyperparams_path = (
            path_to_datafolder
            + f"/{model_name}/"
            + model_name
            + "_"
            + dataset_name
            + "_GridSearchCV-best-hyperparams.csv"
    )
    best_hyperparams = csv_to_dict(hyperparams_path)
    clf = models.ProjectedQuantumKernel(**best_hyperparams)
    clf.initialize(n_features=2)
    grid_flat_trans = clf.transform(grid_flat)
    zero_point_trans = clf.transform(zero_point)
    Z_flat = np.squeeze(clf.precompute_kernel(zero_point_trans, grid_flat_trans))
    Z_flat = (Z_flat - Z_flat.min()) / (Z_flat.max() - Z_flat.min())
    Z = Z_flat.reshape((len(X), len(Y)))

    ax = fig.add_subplot(1, 5, 5, projection="3d")
    ax.set_zorder(2)
    surf = ax.plot_surface(
        X_mesh, Y_mesh, Z, cmap=cmap, linewidth=0, antialiased=False
    )
    ax.set_title(f"Projected\n Quantum\n Kernel")

    plt.suptitle(displayname)
    plt.tight_layout()
    plt.savefig(f"figures/kernel-comparison-{displayname}-2d.png", bbox_inches="tight")
    plt.show()
