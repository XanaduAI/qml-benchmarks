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
Script to reproduce Fig. 10 in the plots, showing the aggregated rankings of models.
"""
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from collections import Counter

sns.set(font_scale=1.3)
sns.set_style("white")

# Plot settings

datasets = {
    "mnist_pca": [
        "../results/mnist_pca",
        "mnist_3-5_",
        "d_GridSearchCV-best-hyperparams-results.csv",
        "number of features",
    ],
    "mnist250": [
        "../results/mnist_pca-",
        "mnist_3-5_",
        "d-250_GridSearchCV-best-hyperparams-results.csv",
        "number of features",
    ],
    "mnist_pixels": [
        "../results/mnist_cg",
        "mnist_pixels_3-5_",
        "_GridSearchCV-best-hyperparams-results.csv",
        "size of image",
    ],
    "bars_and_stripes": [
        "../results/bars_and_stripes",
        "bars_and_stripes_",
        "_GridSearchCV-best-hyperparams-results.csv",
        "size of image",
    ],
    "linearly-separable": [
        "../results/linearly_separable",
        "linearly_separable_",
        "d_GridSearchCV-best-hyperparams-results.csv",
        "number of features",
    ],
    "hmm": [
        "../results/hidden_manifold",
        "hidden_manifold-6manifold-",
        "d_GridSearchCV-best-hyperparams-results.csv",
        "number of features",
    ],
    "hmm-diff": [
        "../results/hidden_manifold_diff",
        "hidden_manifold-10d-",
        "manifold_GridSearchCV-best-hyperparams-results.csv",
        "number of manifolds",
    ],
    "two-manifold": [
        "../results/two_curves",
        "two_curves-5degree-0.1offset-",
        "d_GridSearchCV-best-hyperparams-results.csv",
        "number of features",
    ],
    "two-manifold-diff": [
        "../results/two_curves_diff",
        "two_curves-10d-",
        "degree_GridSearchCV-best-hyperparams-results.csv",
        "degree",
    ],
    "hyperplanes-diff": [
        "../results/hyperplanes_diff",
        "hyperplanes-10d-from3d-",
        "n_GridSearchCV-best-hyperparams-results.csv",
        "number of hyperplanes",
    ],
}

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
    # "SeparableVariationalClassifier"
]

clfs_kernel = [
    "IQPKernelClassifier",
    "ProjectedQuantumKernel",
    "QuantumKitchenSinks",
    "SVC",
    # "SeparableKernelClassifier",
]

clfs_cnn = ["ConvolutionalNeuralNetwork", "WeiNet", "QuanvolutionalNeuralNetwork"]

clfs = [clfs_qnn, clfs_kernel, clfs_cnn]

##############################################################################

with open("plotting_standards.yaml", "r") as stream:
    plotting_config = yaml.safe_load(stream)

colors = plotting_config["color"]
dashes = plotting_config["dashes"]
dashes = {k: eval(v) for k, v in dashes.items()}
markers = plotting_config["marker"]

fig, axes = plt.subplots(3, 1, sharex=False, height_ratios=[9, 5, 3], figsize=(7, 6))

for clf_family, ax in zip(clfs, axes):

    df = pd.DataFrame(columns=["dataset", "test_acc", "train_acc", "Model"])

    for clf in clf_family:
        df_cls = pd.DataFrame(columns=["dataset", "test_acc", "train_acc", "Model"])

        for dataset, (
            path_to_datafolder,
            dataset_name0,
            dataset_name1,
            xlabel,
        ) in datasets.items():

            for n in range(2, 33):

                # load results
                dataset_name = dataset_name0 + str(n) + dataset_name1
                if dataset == "mnist_pixels":
                    dataset_name = dataset_name0 + f"{n}x{n}" + dataset_name1
                if dataset == "bars_and_stripes":
                    dataset_name = dataset_name0 + f"{n}_x_{n}" + dataset_name1

                path_to_results = (
                    path_to_datafolder + f"/{clf}/" + clf + "_" + dataset_name
                )
                try:
                    df_new = pd.read_csv(path_to_results, index_col=0)
                except:
                    continue

                # write into new dataframe and add
                df_new["Model"] = [clf] * len(df_new.index)
                df_new["dataset"] = [str(dataset_name)] * len(df_new.index)
                df_cls = pd.concat([df_cls, df_new])

            df = pd.concat([df, df_cls])

        # average accuracy over individual runs
        df = df.groupby(["dataset", "Model"]).mean()
        df["rank"] = df.groupby("dataset")["test_acc"].rank(
            method="min", ascending=False
        )
        df["rank_pct"] = df.groupby("dataset")["test_acc"].rank(
            method="min", ascending=False, pct=True
        )
        df = df.reset_index()

    # collect statistics about rankings for each model
    stats = {}
    order_score = {}
    for model in clf_family:
        ranks = df[df["Model"] == model]["rank"].to_list()
        ranks_pct = df[df["Model"] == model]["rank_pct"].to_list()

        counts = dict(Counter(ranks))
        total = sum([c for _, c in counts.items()])

        stats[model] = {}
        for i in range(1, len(clf_family) + 1):
            if i in counts:
                stats[model]["rank " + str(i)] = counts[i]
            else:
                stats[model]["rank " + str(i)] = 0

        order_score[model] = np.mean(ranks_pct)

    df_plot = pd.DataFrame(stats).transpose()
    df_plot = df_plot[df_plot.columns[::-1]]

    # identify order (ranking of ranking results)
    df_plot["average_pct"] = df_plot.index.map(lambda model: order_score[model])
    df_plot = df_plot.sort_values(axis=0, by="average_pct", ascending=False)
    df_plot = df_plot.drop(["average_pct"], axis=1)

    df_plot.plot.barh(
        ax=ax,
        stacked=True,
        cmap="coolwarm_r",
        width=0.8,
        legend=False,
        edgecolor="None",
    )

sns.despine()

plt.xlabel("number of rankings")
plt.tight_layout()
plt.savefig(f"figures/ranking-all.png")
plt.show()
