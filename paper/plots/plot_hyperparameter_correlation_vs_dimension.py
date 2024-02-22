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
Script to reproduce Fig. 9 in the plots,
showing the correlation of hyperparameters with the test accuracy for different
dimensions.
"""

import shutil
import os
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)
from matplotlib.ticker import MaxNLocator
from scipy.stats import spearmanr
from qml_benchmarks.hyperparameter_settings import hyper_parameter_settings

sns.set(rc={"figure.figsize": (5.5, 4)})
sns.set(font_scale=1.3)
sns.set_style("white")


def remove_files(folder_path):
    """_summary_

    Args:
        folder_path (_type_): _description_
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            # Check if it is a file and not a directory
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  # Remove the file
            elif os.path.isdir(file_path):
                # Optionally, if you want to remove directories as well
                # Use shutil.rmtree() to remove an entire directory tree
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


classifiers = list(hyper_parameter_settings.keys())
all_hps = []

for classifier in classifiers:
    all_hps += list(hyper_parameter_settings[classifier].keys())

all_hps = set(all_hps)


def string_to_color(s):
    # Generate a hash value
    hash_value = hash(s)

    # Convert the hash to RGB values
    # We use bitwise AND operation to get different parts of the hash
    red = (hash_value & 0xFF0000) >> 16
    green = (hash_value & 0x00FF00) >> 8
    blue = hash_value & 0x0000FF

    return f"#{red:02X}{green:02X}{blue:02X}"


##############################################################################

with open("plotting_standards.yaml", "r") as stream:
    plotting_config = yaml.safe_load(stream)

colors = plotting_config["color"]
colors["SVC"] = "lightgray"
dashes = plotting_config["dashes"]
dashes = {k: eval(v) for k, v in dashes.items()}
markers = plotting_config["marker"]

"""Get correlations between accuracy and different hyperparameters across
all results"""

classifiers = list(hyper_parameter_settings.keys())
all_hps = []

for classifier in classifiers:
    all_hps += list(hyper_parameter_settings[classifier].keys())

all_hps = set(all_hps)


def get_hyperparams_from_cols(
    df: pd.DataFrame, ignore: list[str] = ["batch_size", "max_vmap"]
) -> list[str]:
    """
    Get the hyperparameter names from a dataframe.

    Args:
        df: Pandas dataframe.
    """
    hyperparameter_cols = []

    for name in df.columns:
        if "param_" in name:
            param_name = name.split("param_")[1]
            if param_name in ignore:
                continue
            hyperparameter_cols.append(param_name)
    return hyperparameter_cols


def get_correlation(df):
    """Get Spearman's correlation coefficient for the data"""
    hyperparams = get_hyperparams_from_cols(df)
    df = df[["param_" + col for col in hyperparams] + ["mean_test_accuracy"]]
    spearman_corr, _ = spearmanr(df)
    corr = spearman_corr[:, -1][: len(hyperparams)]
    corr_dict = {}
    for col, val in zip(hyperparams, corr):
        corr_dict[col] = val
    return corr


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


def get_correlations(clf):
    """Get all correlations for dataset

    Args:
        clf (_type_): _description_

    Returns:
        _type_: _description_
    """
    correlations_across_all_model_data = {}
    for dataname, (
        path_to_datafolder,
        dataset_name0,
        dataset_name1,
    ) in datasets.items():
        for n_features in range(2, 21):
            dataset_name = dataset_name0 + str(n_features) + dataset_name1
            path_to_results = path_to_datafolder + f"/{clf}/" + clf + "_" + dataset_name

            try:
                df = pd.read_csv(
                    path_to_results.split("GridSearchCV")[0] + "GridSearchCV.csv"
                )
                df = df.dropna()
                corr = get_correlation(df)
                hyperparams = get_hyperparams_from_cols(df)
                for i, hp in enumerate(hyperparams):
                    correlations_across_all_model_data[
                        (clf, dataname, n_features, hp)
                    ] = corr[i]
            except:
                continue

    return correlations_across_all_model_data


data_list = []
for clf in clfs_qnn:
    results = get_correlations(clf)
    for key in results:
        data_list.append(list(key) + [results[key]])

for clf in clfs_kernel:
    results = get_correlations(clf)
    for key in results:
        data_list.append(list(key) + [results[key]])

for clf in clfs_cnn:
    results = get_correlations(clf)
    for key in results:
        data_list.append(list(key) + [results[key]])

df = pd.DataFrame(
    data_list, columns=["Model", "Dataset", "Dim", "Hyperparameter", "Corr"]
)


datasets = ["MNIST PCA", "HIDDEN MANIFOLD DIFF", "LINEARLY SEPARABLE"]

colors = {
    "MNIST PCA": "black",
    "HIDDEN MANIFOLD DIFF": "blue",
    "LINEARLY SEPARABLE": "orange",
}

markers = {"MNIST PCA": "o", "HIDDEN MANIFOLD DIFF": "x", "LINEARLY SEPARABLE": "s"}

hyperparameters = ["n_layers"]

plt.figure()
for dataset in datasets:
    for hyperparam in hyperparameters:
        aggregated_data = []
        for dim in sorted(df["Dim"].unique()):
            # Filter data for current dataset, hyperparameter, and dimension
            dim_data = df[
                (df["Dim"] == dim)
                & (df["Dataset"] == dataset)
                & (df["Hyperparameter"] == hyperparam)
            ]
            # Calculate mean and std of Corr across all classifiers for the current dimension
            mean_corr = dim_data["Corr"].mean()
            std_corr = dim_data["Corr"].std()
            aggregated_data.append((dim, mean_corr, std_corr))

        # Convert aggregated data to a DataFrame
        aggregated_df = pd.DataFrame(
            aggregated_data, columns=["Dim", "Mean_Corr", "Std_Corr"]
        )

        # Plot the mean Correlation as a line
        plt.plot(
            aggregated_df["Dim"],
            aggregated_df["Mean_Corr"],
            color=colors[dataset],
            marker=markers[dataset],
            label=dataset,
        )

        # Add shaded area for std deviation
        plt.fill_between(
            aggregated_df["Dim"],
            aggregated_df["Mean_Corr"] - aggregated_df["Std_Corr"],
            aggregated_df["Mean_Corr"] + aggregated_df["Std_Corr"],
            alpha=0.2,
            color=colors[dataset],
        )

plt.ylim(-1, 1)
plt.xlim(2, 20)
plt.xlabel("dimension")
plt.ylabel("correlation")
plt.legend(loc="lower left")
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
# Show the plot
plt.grid()
plt.tight_layout()
plt.savefig(f"figures/hyperparameter_correlation_vs_dimension.png", bbox_inches="tight")
plt.show()
