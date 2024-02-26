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
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay
from qml_benchmarks import models
from qml_benchmarks.hyperparam_search_utils import read_data

os.makedirs("figures", exist_ok=True)

sns.set(rc={"figure.figsize": (6, 6)})
sns.set(font_scale=1.3)
sns.set_style("white")
palette = sns.color_palette("deep")
cmap = ListedColormap(palette)


def csv_to_dict(file_path):
    dict = {}
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        # Skip the first line
        next(csvreader)
        for row in csvreader:
            hyperparameter, value = row
            # Check if the value is numeric and convert it to int or float accordingly
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass  # If conversion is not possible, keep the value as a string
            dict[hyperparameter] = value
    return dict


X_train, y_train = read_data(
    "datasets-for-plots/linearly_separable/linearly_separable_2d_train.csv"
)
X_test, y_test = read_data(
    "datasets-for-plots/linearly_separable/linearly_separable_2d_test.csv"
)

figure = plt.figure()

# just plot the dataset first
cmap = sns.diverging_palette(30, 255, l=60, as_cmap=True)
ax = plt.subplot(1, 1, 1)

models_to_plot = [
    models.QuantumMetricLearner,
    models.IQPKernelClassifier,
    models.IQPVariationalClassifier,
    models.SeparableVariationalClassifier,
]
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)

for ax, model in zip([ax1, ax2, ax3, ax4], models_to_plot):

    clf_name = model.__name__
    hyperparams_path = f"../results/linearly_separable/{clf_name}/{clf_name}_linearly_separable_2d_GridSearchCV-best-hyperparams.csv"
    best_hyperparams = csv_to_dict(hyperparams_path)
    clf = model(**best_hyperparams)
    clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)
    DecisionBoundaryDisplay.from_estimator(
        clf, X_train, cmap=cmap, alpha=0.8, ax=ax, eps=0.5
    )

    # Plot the training points
    ax.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        cmap=cmap,
        marker="o",
        edgecolors="k",
    )
    # Plot the testing points
    ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap=cmap,
        edgecolors="k",
        marker="^",
        alpha=1.0,
    )

    ax.set_title(f"{clf_name}")

plt.tight_layout()
plt.savefig(
    f"figures/2d-linsep-decisionboundaries.png", bbox_inches="tight"
)
plt.show()
