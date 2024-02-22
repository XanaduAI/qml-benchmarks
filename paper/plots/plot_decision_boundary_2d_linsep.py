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
Script to reproduce Fig. 19 in the plots, showing the decision boundaries of selected models
on the 2d linearly separable dataset.
"""
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay
from qml_benchmarks import models
from qml_benchmarks.hyperparam_search_utils import read_data

sns.set(rc={"figure.figsize": (3, 3)})
sns.set(font_scale=1.5)
sns.set_style("white")


X_train, y_train = read_data(
    "../benchmarks/linearly_separable/linearly_separable_2d_train.csv"
)
X_test, y_test = read_data(
    "../benchmarks/linearly_separable/linearly_separable_2d_test.csv"
)

figure = plt.figure()

# just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(["#FF0000", "#0000FF"])
ax = plt.subplot(1, 1, 1)

for model in [
    models.QuantumMetricLearner,
    models.IQPKernelClassifier,
    models.IQPVariationalClassifier,
    models.SeparableVariationalClassifier,
]:

    clf_name = model.__name__
    hyperparams_path = f"../results/linearly_separable/{clf_name}/{clf_name}_linearly_separable_2d_GridSearchCV-best-hyperparams.pickle"
    best_hyperparams = pickle.load(open(hyperparams_path, "rb"))
    clf = model(**best_hyperparams)
    clf.fit(X_train, y_train)

    ax = plt.subplot(1, 1, 1)
    score = clf.score(X_test, y_test)
    DecisionBoundaryDisplay.from_estimator(
        clf, X_train, cmap=cm, alpha=0.8, ax=ax, eps=0.5
    )

    # Plot the training points
    ax.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        cmap=cm_bright,
        marker="o",
        edgecolors="k",
    )
    # Plot the testing points
    ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap=cm_bright,
        edgecolors="k",
        marker="^",
        alpha=0.6,
    )
    plt.title(f"{clf_name}")
    plt.savefig(
        f"figures/{clf_name}-2d-linsep-decisionboundary.png", bbox_inches="tight"
    )
    plt.show()
