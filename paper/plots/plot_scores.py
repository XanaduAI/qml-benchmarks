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
Script to reproduce Figs. 11 and 24 in the plots, showing the test accuracies of models.
"""
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.ticker import MaxNLocator

sns.set(rc={"figure.figsize": (8, 4)})
sns.set(font_scale=1.3)
sns.set_style("white")

datasets = {
    "MNIST PCA": [
        "../results/mnist_pca",
        "mnist_3-5_",
        "d_GridSearchCV-best-hyperparams-results.csv",
        "number of features",
    ],
    "MNIST PCA-": [
        "../results/mnist_pca-",
        "mnist_3-5_",
        "d-250_GridSearchCV-best-hyperparams-results.csv",
        "number of features",
    ],
    "MNIST CG": [
        "../results/mnist_cg",
        "mnist_pixels_3-5_",
        "_GridSearchCV-best-hyperparams-results.csv",
        "size of image",
    ],
    "BARS & STRIPES": [
        "../results/bars_and_stripes",
        "bars_and_stripes_",
        "_0.5noise_GridSearchCV-best-hyperparams-results.csv",
        "size of image",
    ],
    "LINEARLY SEPARABLE": [
        "../results/linearly_separable",
        "linearly_separable_",
        "d_GridSearchCV-best-hyperparams-results.csv",
        "number of features",
    ],
    "HIDDEN MANIFOLD": [
        "../results/hidden_manifold",
        "hidden_manifold-6manifold-",
        "d_GridSearchCV-best-hyperparams-results.csv",
        "number of features",
    ],
    "HIDDEN MANIFOLD DIFF": [
        "../results/hidden_manifold_diff",
        "hidden_manifold-10d-",
        "manifold_GridSearchCV-best-hyperparams-results.csv",
        "number of manifolds",
    ],
    "TWO CURVES": [
        "../results/two_curves_diff",
        "two_curves-5degree-0.1offset-",
        "d_GridSearchCV-best-hyperparams-results.csv",
        "number of features",
    ],
    "TWO CURVES DIFF": [
        "../results/two_curves_diff",
        "two_curves-10d-",
        "degree_GridSearchCV-best-hyperparams-results.csv",
        "degree",
    ],
    "HYPERPLANES DIFF": [
        "../results/hyperplanes_diff",
        "hyperplanes-10d-from3d-",
        "n_GridSearchCV-best-hyperparams-results.csv",
        "number of hyperplanes",
    ],
}

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
    "SVC",
    "IQPKernelClassifier",
    "ProjectedQuantumKernel",
    "QuantumKitchenSinks",
]

clfs_cnn = ["ConvolutionalNeuralNetwork", "WeiNet", "QuanvolutionalNeuralNetwork"]

clfs_custom = [
    "IQPVariationalClassifier",
    "IQPKernelClassifier",
]

with open("plotting_standards.yaml", "r") as stream:
    plotting_config = yaml.safe_load(stream)

colors = plotting_config["color"]
dashes = plotting_config["dashes"]
dashes = {k: eval(v) for k, v in dashes.items()}
markers = plotting_config["marker"]

for clfs, out_name in [(clfs_qnn, "qnn"), (clfs_kernel, "kernel"), (clfs_cnn, "cnn")]:

    for dataname, (
        path_to_datafolder,
        dataset_name0,
        dataset_name1,
        xlabel,
    ) in datasets.items():

        df = pd.DataFrame(columns=["n_features", "test_acc", "train_acc", "clf"])
        df = df.rename(columns={"clf": "Model"})

        for clf in clfs:

            df_cls = pd.DataFrame(columns=["n", "test_acc", "train_acc", "Model"])

            for n in range(2, 33):

                dataset_name = dataset_name0 + str(n) + dataset_name1
                if dataname == "MNIST CG":
                    dataset_name = dataset_name0 + f"{n}x{n}" + dataset_name1
                if dataname == "BARS & STRIPES":
                    dataset_name = dataset_name0 + f"{n}_x_{n}" + dataset_name1

                path_to_results = (
                    path_to_datafolder + f"/{clf}/" + clf + "_" + dataset_name
                )

                try:
                    df_new = pd.read_csv(path_to_results)
                except:
                    print(f"{path_to_results} not found")

                    continue

                df_new["Model"] = [clf] * len(df_new.index)
                df_new["n"] = [n] * len(df_new.index)
                df_cls = pd.concat([df_cls, df_new])

            df = pd.concat([df, df_cls])

        if len(df.index) == 0:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True, tight_layout=True)
        axes[0].set_title("train")
        axes[1].set_title("test")

        sns.lineplot(
            ax=axes[0],
            data=df,
            x="n",
            y="train_acc",
            hue="Model",
            palette=colors,
            markers=markers,
            dashes=dashes,
            style="Model",
        )

        sns.lineplot(
            ax=axes[1],
            data=df,
            x="n",
            y="test_acc",
            hue="Model",
            palette=colors,
            markers=markers,
            dashes=dashes,
            style="Model",
        )
        sns.despine()

        for i in [0, 1]:
            axes[i].xaxis.set_major_locator(MaxNLocator(integer=True))
            axes[i].set_ylim((0.45, 1.05))
            axes[i].set_ylabel("accuracy")
            axes[i].set_xlabel(xlabel)

        fig.suptitle(f"{dataname}", fontsize=15, y=0.9)

        axes[0].get_legend().remove()
        axes[1].get_legend().remove()
        axes[0].grid(axis="y")
        axes[1].grid(axis="y")
        plt.tight_layout()
        plt.savefig(f"figures/score-{dataname}-{out_name}.png")
        plt.show()
