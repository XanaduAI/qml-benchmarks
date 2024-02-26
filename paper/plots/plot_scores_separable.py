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
Script to reproduce Figs. 12 and 25 in the plots, showing the test accuracies of separable models.
"""
import pandas as pd
import seaborn as sns
import yaml
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)
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
    "SeparableVariationalClassifier",
]

clfs_kernel = [
    "SVC",
    "IQPKernelClassifier",
    "ProjectedQuantumKernel",
    "QuantumKitchenSinks",
    "SeparableKernelClassifier",
]

clfs_cnn = ["ConvolutionalNeuralNetwork", "WeiNet", "QuanvolutionalNeuralNetwork"]

color = {
    "CircuitCentricClassifier": "darkgray",
    "DataReuploadingClassifier": "darkgray",
    "DressedQuantumCircuitClassifier": "darkgray",
    "IQPVariationalClassifier": "darkgray",
    "QuantumMetricLearner": "darkgray",
    "ClassicalFourierMPS": "black",
    "MLPClassifier": "black",
    "Perceptron": "black",
    "QuanvolutionalNeuralNetwork": "darkgray",
    "WeiNet": "darkgray",
    "ConvolutionalNeuralNetwork": "black",
    "SVC": "black",
    "IQPKernelClassifier": "darkgray",
    "ProjectedQuantumKernel": "darkgray",
    "QuantumKitchenSinks": "darkgray",
    "QuantumBoltzmannMachine": "darkgray",
    "SVClinear": "black",
    "ParallelGradients": "darkgray",
    "TreeTensorClassifier": "darkgray",
    "SeparableVariationalClassifier": "teal",
    "SeparableKernelClassifier": "deeppink",
}


##############################################################################

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

            for n_features in range(2, 21):

                dataset_name = dataset_name0 + str(n_features) + dataset_name1
                path_to_results = (
                    path_to_datafolder + f"/{clf}/" + clf + "_" + dataset_name
                )

                try:
                    df_new = pd.read_csv(path_to_results)
                except:
                    print(f"{path_to_results} not found")

                    continue

                df_new["Model"] = [clf] * len(df_new.index)
                df_new["n"] = [n_features] * len(df_new.index)
                df_cls = pd.concat([df_cls, df_new])

            df = pd.concat([df, df_cls])

        if len(df.index) == 0:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
        ax.set_title("test")

        sns.lineplot(
            ax=ax,
            data=df,
            x="n",
            y="test_acc",
            hue="Model",
            palette=color,
            markers=markers,
            dashes=dashes,
            style="Model",
        )
        sns.despine()

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim((0.45, 1.05))
        ax.set_ylabel("accuracy")
        ax.set_xlabel(xlabel)

        fig.suptitle(f"{dataname}", fontsize=15)

        ax.grid(axis="y")
        ax.get_legend().remove()

        ## hack to plot legend in an svg and hand-extract.
        ## Need to comment out the removal of the second legend above and saving below.
        # plt.savefig(f"figures/score-separable-{model_family}-legend.svg")

        plt.tight_layout()
        plt.savefig(f"figures/score-separable-{dataname}-{out_name}.png")

        plt.show()
