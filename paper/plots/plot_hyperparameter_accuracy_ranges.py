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
Script to reproduce Fig. 6 in the plots,
showing the variation in accuracy as hyperparameters are changed.
"""
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)
from matplotlib.ticker import MaxNLocator, MultipleLocator

sns.set(font_scale=1.4)
sns.set_style("white")

datasets = {
    "MNIST PCA": [
        "../results/mnist_pca",
        "mnist_3-5_",
        "d_GridSearchCV-best-hyperparams-results.csv",
    ],
    "MNIST PCA-": [
        "../results/mnist_pca-",
        "mnist_3-5_",
        "d-250_GridSearchCV-best-hyperparams-results.csv",
    ],
    "LINEARLY SEPARABLE": [
        "../results/linearly_separable",
        "linearly_separable_",
        "d_GridSearchCV-best-hyperparams-results.csv",
    ],
    "HIDDEN MANIFOLD": [
        "../results/hidden_manifold",
        "hidden_manifold-6manifold-",
        "d_GridSearchCV-best-hyperparams-results.csv",
    ],
    "HIDDEN MANIFOLD DIFF": [
        "../results/hidden_manifold_diff",
        "hidden_manifold-10d-",
        "manifold_GridSearchCV-best-hyperparams-results.csv",
    ],
    "TWO CURVES": [
        "../results/two_curves",
        "two_curves-5degree-0.1offset-",
        "d_GridSearchCV-best-hyperparams-results.csv",
    ],
    "TWO CURVES DIFF": [
        "../results/two_curves_diff",
        "two_curves-10d-",
        "degree_GridSearchCV-best-hyperparams-results.csv",
    ],
    "HYPERPLANES DIFF": [
        "../results/hyperplanes_diff",
        "hyperplanes-10d-from3d-",
        "n_GridSearchCV-best-hyperparams-results.csv",
    ],
}

# Plot settings #################################

# models to plot
clfs_qnn = [
    "MLPClassifier",
    "CircuitCentricClassifier",
    "DataReuploadingClassifier",
    "DressedQuantumCircuitClassifier",
    "IQPVariationalClassifier",
    "QuantumMetricLearner",
    "QuantumBoltzmannMachine",
    "TreeTensorClassifier",
]

clfs_kernel = [
    "IQPKernelClassifier",
    "ProjectedQuantumKernel",
    "QuantumKitchenSinks",
    "SVC",
]

clfs_cnn = ["ConvolutionalNeuralNetwork", "WeiNet", "QuanvolutionalNeuralNetwork"]
##############################################################################

with open("plotting_standards.yaml", "r") as stream:
    plotting_config = yaml.safe_load(stream)

colors = plotting_config["color"]
colors["SVC"] = "lightgray"
dashes = plotting_config["dashes"]
dashes = {k: eval(v) for k, v in dashes.items()}
markers = plotting_config["marker"]


def plot_data(ax, dataname, clf):
    """Plot the spread of hyperparameters for given datasets and classifiers

    Args:
        datasets (_type_): _description_
        clfs (_type_): _description_
    """
    path_to_datafolder, dataset_name0, dataset_name1 = datasets[dataname]
    df = pd.DataFrame(
        columns=[
            "n_features",
            "test_acc",
            "train_acc",
            "clf",
            "min_test_accuracy",
            "max_test_accuracy",
        ]
    )
    df = df.rename(columns={"clf": "Model"})
    df_cls = pd.DataFrame(
        columns=[
            "n",
            "test_acc",
            "train_acc",
            "Model",
            "min_test_accuracy",
            "max_test_accuracy",
        ]
    )
    for n_features in range(2, 21):

        dataset_name = dataset_name0 + str(n_features) + dataset_name1
        path_to_results = path_to_datafolder + f"/{clf}/" + clf + "_" + dataset_name

        try:
            df_new = pd.read_csv(path_to_results)
        except:
            print(f"{path_to_results} not found")
            continue
        df_new_grid = pd.read_csv(
            path_to_results.split("GridSearchCV")[0] + "GridSearchCV.csv"
        )
        try:
            max_test, min_test = np.max(df_new_grid["mean_test_accuracy"]), np.min(
                df_new_grid["mean_test_accuracy"]
            )
        except KeyError:
            max_test, min_test = np.max(df_new_grid["mean_test_score"]), np.min(
                df_new_grid["mean_test_score"]
            )

        df_new["Model"] = [clf] * len(df_new.index)
        df_new["n"] = [n_features] * len(df_new.index)

        df_new["min_test_accuracy"] = [min_test] * len(df_new.index)
        df_new["max_test_accuracy"] = [max_test] * len(df_new.index)

        df_cls = pd.concat([df_cls, df_new])

    df = pd.concat([df, df_cls])
    models = df["Model"].unique()

    # Iterate over each model to fill the area between min_test_accuracy and max_test_accuracy
    for model in models:
        model_df = df[df["Model"] == model]
        ax.fill_between(
            model_df["n"].values.astype(float),
            model_df["min_test_accuracy"],
            model_df["max_test_accuracy"],
            alpha=0.3,
            facecolor=colors[model],
        )
    sns.despine()


datasets_to_plot = ["LINEARLY SEPARABLE", "MNIST PCA", "HIDDEN MANIFOLD DIFF"]
xlabels = ["# features", "# features", "# manifolds"]
classifiers_to_plot = [
    "MLPClassifier",
    "DataReuploadingClassifier",
    "CircuitCentricClassifier",
    "IQPVariationalClassifier",
]

# Calculate figure size: each subplot is 5x4, so the figure's width and height should be adjusted accordingly
fig_width = 8
fig_height = 10
fig, axs = plt.subplots(
    len(classifiers_to_plot),
    len(datasets_to_plot),
    figsize=(fig_width, fig_height),
    sharex="col",
    sharey="row",
)

for col_index, dataname in enumerate(datasets_to_plot):

    for row_index, clf in enumerate(classifiers_to_plot):
        ax = axs[row_index, col_index]

        # plot the range of accuracies
        plot_data(ax, dataname, clf)

        # Setting titles for the top row and y-labels for the first column
        if row_index == 0:
            if len(dataname) > 15:
                ax.set_title(
                    (
                        " ".join(dataname.split(" ")[:-1])
                        + "\n"
                        + " "
                        + "".join(dataname.split(" ")[-1])
                    ),
                )
            else:
                ax.set_title(dataname)
        if col_index == 0:
            ax.set_ylabel(clf.split("Classifier")[0] + "\naccuracy")

        # Customizing each subplot (ax) as needed, e.g., set limits, labels, etc.
        ax.set_ylim(0.45, 1.05)
        ax.grid(axis="y")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MultipleLocator(5))

        ax.set_xlim(2, 20)
        if row_index == len(classifiers_to_plot) - 1:
            ax.set_xlabel(xlabels[col_index])

plt.tight_layout()
plt.subplots_adjust(wspace=0.07)
plt.savefig("figures/hyperparameter_accuracy_ranges.png", bbox_inches="tight")
plt.show()
