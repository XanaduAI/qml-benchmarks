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
Script to reproduce Fig. 2 in the plots, showing the effect of data on a benchmarking result:
Two different seeds for data generation change the test accuracy from almost random guessing to perfect score.
"""
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
from qml_benchmarks.models import VanillaQNN

sns.set(rc={"figure.figsize": (4, 3)})
sns.set(font_scale=1.3)
sns.set_style("white")

for seed in [42, 44]:

    # make data
    X, y = make_blobs(
        n_features=2,
        center_box=(-1.6, 1.6),
        cluster_std=0.15,
        centers=3,
        random_state=seed,
    )
    y = np.array([-1 if y_ == 0 else 1 for y_ in y])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    # train model
    clf = VanillaQNN(embedding_layers=2)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    pred = clf.predict(X_test)

    # create figure
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    ax = plt.subplot(1, 1, 1)

    # plot the decision regions
    DecisionBoundaryDisplay.from_estimator(clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5)
    # plot the training points
    ax.scatter(
        X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, s=70, edgecolors="k"
    )
    # plot the testing points
    ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap=cm_bright,
        marker="^",
        s=70,
        edgecolors="k",
    )

    # some formatting
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.text(
        x_max,
        y_min - 0.5,
        "test accuracy = %.2f" % score,
        size=18,
        horizontalalignment="right",
    )
    plt.tight_layout()

    plt.savefig(f"figures/effect_of_variation_in_data_seed-{seed}.png")
    plt.show()
