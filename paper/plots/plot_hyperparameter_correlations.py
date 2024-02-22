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
Script to reproduce Fig. 8 in the plots,
showing the correlation of hyperparameters with the test accuracy.
"""
import yaml
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)
import matplotlib.colors as mcolors
from qml_benchmarks.hyperparameter_settings import hyper_parameter_settings

sns.set(rc={"figure.figsize": (5.5, 4)})
sns.set(font_scale=1.2)
sns.set_style("white")

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


colors = {s: string_to_color(s) for s in all_hps}


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

with open("plotting_standards.yaml", "r") as stream:
    plotting_config = yaml.safe_load(stream)

colors = plotting_config["color"]
colors["SVC"] = "lightgray"
dashes = plotting_config["dashes"]
dashes = {k: eval(v) for k, v in dashes.items()}
markers = plotting_config["marker"]


def plot_correlation(clf):
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


def generate_pastel_colors(n):
    """
    Generates n distinct pastel colors.

    :param n: Number of distinct colors to generate.
    :return: List of pastel colors in hex format.
    """
    # Base saturation and value for pastel colors
    base_saturation, base_value = 0.6, 0.9

    # Generate colors
    colors = []
    for i in np.linspace(0, 1, n, endpoint=False):
        # HSV: Hue varies, Saturation and Value are fixed for pastel effect
        hue = i
        saturation = base_saturation + np.random.rand() * (1.0 - base_saturation)
        value = base_value + np.random.rand() * (1.0 - base_value)
        rgb = mcolors.hsv_to_rgb((hue, saturation, value))
        colors.append(mcolors.rgb2hex(rgb))

    return colors


data_list = []
for clf in clfs_qnn:
    results = plot_correlation(clf)
    for key in results:
        data_list.append(list(key) + [results[key]])

for clf in clfs_kernel:
    results = plot_correlation(clf)
    for key in results:
        data_list.append(list(key) + [results[key]])

for clf in clfs_cnn:
    results = plot_correlation(clf)
    for key in results:
        data_list.append(list(key) + [results[key]])

df = pd.DataFrame(
    data_list, columns=["Model", "Dataset", "Dim", "hyperparameter", "Corr"]
)
print(df.head())

# Grouping by 'Hyperparameter' column and calculating mean and standard deviation
group = "hyperparameter"
grouped_data = df.groupby(group)["Corr"].agg(["mean", "std"]).reset_index()

# Sorting by mean correlation
grouped_data = grouped_data.sort_values(by="mean", ascending=True).reset_index()

# Example: Generate a list of colors based on the size of an existing list
colors = {}
generated_colors = generate_pastel_colors(len(grouped_data))

for i, color in enumerate(generated_colors):
    colors[i] = color

hyperparameters_to_show = ["t", "learning_rate_init",
                           "repeats", "temperature",
                           "n_input_copies", "observable_type",
                           "learning_rate", "gamma_factor",
                           "alpha", "hidden_layer_sizes",
                           "trotter_steps", "n_qfeatures",
                           "visible_qubits", "n_layers",
                           "C", "n_episodes"]
grouped_data = grouped_data.loc[grouped_data['hyperparameter'].isin(hyperparameters_to_show)]

# Plotting
fig, ax = plt.subplots()

# For each group, plot a horizontal bar with the mean as center and std as height
for i in range(len(grouped_data)):
    mean = grouped_data.loc[i, "mean"]
    std = grouped_data.loc[i, "std"]
    ax.barh(
        grouped_data[group][i],
        width=2 * std,
        left=mean - std,
        alpha=0.5,
        color=colors[i],
    )

ax.set_ylabel(group)
ax.set_xlabel("average correlation")
ax.set_xlim(-1, 1)
plt.grid()
plt.tight_layout()
plt.savefig("figures/hyperparameter_correlations", bbox_inches="tight")
plt.show()
