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
Script to reproduce Fig. 14 in the plots, showing the feature being transformed by a quantum versus classical
layer in the DressedQuantumCircuitClassifier.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from qml_benchmarks.models import (
    DressedQuantumCircuitClassifier,
    DressedQuantumCircuitClassifierOnlyNN,
)


# create data
X, y = make_moons(n_samples=200, noise=0.1)
y = np.array([-1 if y_ == 0 else 1 for y_ in y])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

cmap = sns.diverging_palette(30, 255, l=60, as_cmap=True)

# original model
clf = DressedQuantumCircuitClassifier()
clf.fit(X_train, y_train)

X_train_trans = clf.transform(X_train)
X_train_after_first_nn = np.array(
    [clf.input_transform(clf.params_, x) for x in X_train_trans]
)
X_train_after_qnn = np.array(
    [np.array(clf.circuit(clf.params_, x)).T for x in X_train_after_first_nn]
)
X_train_after_second_nn = np.array(
    [clf.output_transform(clf.params_, x) for x in X_train_after_qnn]
)

fig, axes = plt.subplots(1, 4, figsize=(8, 2))

axes[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, s=8)
axes[0].set_title("original\n training data")
axes[1].scatter(
    X_train_after_first_nn[:, 0],
    X_train_after_first_nn[:, 1],
    c=y_train,
    cmap=cmap,
    s=8,
)
axes[1].set_title("after first\n NN layer")
axes[2].scatter(
    X_train_after_qnn[:, 0], X_train_after_qnn[:, 1], c=y_train, cmap=cmap, s=8
)
axes[2].set_title("after\n quantum layer")
axes[3].scatter(
    X_train_after_second_nn[:, 0],
    X_train_after_second_nn[:, 1],
    c=y_train,
    cmap=cmap,
    s=8,
)
axes[3].set_title("after second\n nn layer")

sns.despine()
plt.tight_layout()
plt.savefig("figures/feature-transformations-dqcc.png")
plt.show()

# altered model

clf = DressedQuantumCircuitClassifierOnlyNN()
clf.fit(X_train, y_train)

X_train_trans = clf.transform(X_train)
X_train_after_first_nn = np.array(
    [clf.input_transform(clf.params_, x) for x in X_train_trans]
)
X_train_after_mid_nn = np.array(
    [clf.mid_transform(clf.params_, x) for x in X_train_after_first_nn]
)
X_train_after_second_nn = np.array(
    [clf.output_transform(clf.params_, x) for x in X_train_after_mid_nn]
)

fig, axes = plt.subplots(1, 4, figsize=(8, 2))

axes[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, s=8)
axes[0].set_title("original\n training data")
axes[1].scatter(
    X_train_after_first_nn[:, 0],
    X_train_after_first_nn[:, 1],
    c=y_train,
    cmap=cmap,
    s=8,
)
axes[1].set_title("after first\n NN layer")
axes[2].scatter(
    X_train_after_mid_nn[:, 0],
    X_train_after_mid_nn[:, 1],
    c=y_train,
    cmap=cmap,
    s=8,
)
axes[2].set_title("after second\n  NN layer")
axes[3].scatter(
    X_train_after_second_nn[:, 0],
    X_train_after_second_nn[:, 1],
    c=y_train,
    cmap=cmap,
    s=8,
)
axes[3].set_title("after third\n nn layer")

sns.despine()

plt.tight_layout()
plt.savefig("figures/feature-transformations-dqcc-nn.png")
plt.show()
