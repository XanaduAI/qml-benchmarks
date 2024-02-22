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
Script to reproduce Fig. 17 from the plots, comparing different Gram matrices.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)
import seaborn as sns
from qml_benchmarks import models
from qml_benchmarks.hyperparam_search_utils import read_data

sns.set(rc={"figure.figsize": (6, 6)})
sns.set(font_scale=1.0)
sns.set_style("white")


def rbf_kernel(X1, X2, gamma):
    K = np.zeros(shape=(len(X1), len(X2)))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            K[i, j] = np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)
    return K


def linear_kernel(X1, X2):
    K = np.zeros(shape=(len(X1), len(X2)))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            K[i, j] = np.dot(x1, x2)
    return K


def alignment(gram1, gram2):
    """Rescale the gram matrix entries to lie in interval [0, 1]
    and compute their alignment"""
    n_entries = gram1.shape[0] ** 2
    gram1 = (gram1 - gram1.min()) / (gram1.max() - gram1.min())
    gram2 = (gram2 - gram2.min()) / (gram2.max() - gram2.min())
    return np.sum((gram1 - gram2) ** 2) / n_entries


datasets = {
    "mnist_pca": [
        "../benchmarks/mnist_pca-/",
        "../results/mnist_pca-",
        "mnist_3-5_",
        "d-250",
        "MNIST PCA-",
    ],
    "linearly-separable": [
        "../benchmarks/linearly_separable/",
        "../results/linearly-separable",
        "linearly_separable_",
        "d",
        "LINEARLY SEPARABLE",
    ],
    "hmm": [
        "../benchmarks/hidden_manifold/",
        "../results/hidden_manifold",
        "hidden_manifold-6manifold-",
        "d",
        "HIDDEN MANIFOLD",
    ],
    "hmm-diff": [
        "../benchmarks/hidden_manifold/",
        "../results/hidden_manifold_diff",
        "hidden_manifold-",
        "d-10manifold",
        "HIDDEN MANIFOLD DIFF",
    ],
    "two-curves": [
        "../benchmarks/two_curves_diff/",
        "../results/two_curves_diff",
        "two_curves-5degree-0.1offset-",
        "d",
        "TWO CURVES",
    ],
    "two-curves-diff": [
        "../benchmarks/two_curves/",
        "../results/hidden_manifold",
        "two_curves-",
        "d-10degree",
        "TWO CURVES DIFF",
    ],
    "hyperplanes-diff": [
        "../benchmarks/hyperplanes_diff/",
        "../results/hyperplanes_diff",
        "hyperplanes-",
        "d-from3d-10n",
        "HYPERPLANES DIFF",
    ],
}

for dataname, (
    path_to_data,
    path_to_results_folder,
    dataset_name0,
    dataset_name1,
    displayname,
) in datasets.items():

    for k in [2, 10]:

        dataset_name = dataset_name0 + str(k) + dataset_name1
        try:
            X, _ = read_data(path_to_data + dataset_name + "_test.csv")
        except FileNotFoundError:
            continue

        # ----------------------

        model_name = "IQPKernelClassifier"
        hyperparams_path = (
            path_to_results_folder
            + f"/{model_name}/"
            + model_name
            + "_"
            + dataset_name
            + "_GridSearchCV-best-hyperparams.pickle"
        )
        file = open(hyperparams_path, "rb")
        best_hyperparams = pickle.load(file)
        clf = models.IQPKernelClassifier(**best_hyperparams)
        clf.initialize(n_features=k)
        X = clf.transform(X)
        gram_iqp = np.squeeze(clf.precompute_kernel(X, X))

        # ----------------------

        model_name = "ProjectedQuantumKernel"
        hyperparams_path = (
            path_to_results_folder
            + f"/{model_name}/"
            + model_name
            + "_"
            + dataset_name
            + "_GridSearchCV-best-hyperparams.pickle"
        )
        file = open(hyperparams_path, "rb")
        best_hyperparams = pickle.load(file)
        clf = models.ProjectedQuantumKernel(**best_hyperparams)
        clf.initialize(n_features=k)
        X = clf.transform(X)
        gram_pqk = np.squeeze(clf.precompute_kernel(X, X))

        # ----------------------

        model_name = "SeparableKernelClassifier"
        hyperparams_path = (
            path_to_results_folder
            + f"/{model_name}/"
            + model_name
            + "_"
            + dataset_name
            + "_GridSearchCV-best-hyperparams.pickle"
        )
        file = open(hyperparams_path, "rb")
        best_hyperparams = pickle.load(file)
        clf = models.SeparableKernelClassifier(**best_hyperparams)
        clf.initialize(n_features=k)
        X = clf.transform(X)
        gram_sep = np.squeeze(clf.precompute_kernel(X, X))

        # ----------------------

        model_name = "SVC"
        hyperparams_path = (
            path_to_results_folder
            + f"/{model_name}/"
            + model_name
            + "_"
            + dataset_name
            + "_GridSearchCV-best-hyperparams.pickle"
        )
        file = open(hyperparams_path, "rb")
        best_hyperparams = pickle.load(file)
        gamma = best_hyperparams["gamma"]
        gram_svc = np.squeeze(rbf_kernel(X, X, gamma))

        # ----------------------

        model_name = "QuantumKitchenSinks"
        hyperparams_path = (
            path_to_results_folder
            + f"/{model_name}/"
            + model_name
            + "_"
            + dataset_name
            + "_GridSearchCV-best-hyperparams.pickle"
        )
        file = open(hyperparams_path, "rb")
        best_hyperparams = pickle.load(file)
        clf = models.QuantumKitchenSinks(**best_hyperparams)
        clf.initialize(n_features=k)
        X = clf.transform(X)
        gram_qks = np.squeeze(linear_kernel(X, X))

        # ----------------------

        # Compute alignment
        kernels = [gram_svc, gram_sep, gram_iqp, gram_pqk, gram_qks]
        names = [
            "SVC",
            "SeparableKernelClassifier",
            "IQPKernelClassifier",
            "ProjectedQuantumKernel",
            "QuantumKitchenSinks",
        ]

        alignments = np.zeros((len(kernels), len(kernels)))
        for i in range(len(kernels)):
            for j in range(len(kernels)):
                alignments[i, j] = alignment(kernels[i], kernels[j])

        fig, ax = plt.subplots()
        plt.tight_layout()
        pos = plt.imshow(alignments, cmap="rainbow", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(names)), labels=names)
        ax.set_yticks(np.arange(len(names)), labels=names)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        fig.colorbar(pos)
        plt.title(f"{displayname} {k}d")
        plt.tight_layout()
        plt.savefig(f"figures/gram-comparison-{displayname}-{k}d.png")
        plt.show()
