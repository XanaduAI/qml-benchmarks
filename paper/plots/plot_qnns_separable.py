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
Script to reproduce Fig. 13 in the plots, comparing the test accuracies of selected models to
their separable versions.
"""
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)
from matplotlib.ticker import MaxNLocator

sns.set(rc={"figure.figsize": (4, 6)})
sns.set(font_scale=1.2)
sns.set_style("white")

datasets = {
    "MNIST PCA": [
        "../results/mnist_pca",
        "mnist_3-5_",
        "d_GridSearchCV-best-hyperparams-results.csv",
        "number of features",
    ],
}

clfs = [
    ["DataReuploadingClassifier", "DataReuploadingClassifierSeparable"],
    ["DressedQuantumCircuitClassifier", "DressedQuantumCircuitClassifierSeparable"],
    ["QuantumBoltzmannMachine", "QuantumBoltzmannMachineSeparable"],
]


with open("plotting_standards.yaml", "r") as stream:
    plotting_config = yaml.safe_load(stream)

# extend colour palettes by new models
colors = plotting_config["color"]
colors.update(
    {
        "DataReuploadingClassifierSeparable": colors["DataReuploadingClassifier"],
        "DressedQuantumCircuitClassifierSeparable": colors[
            "DressedQuantumCircuitClassifier"
        ],
        "QuantumBoltzmannMachineSeparable": colors["QuantumBoltzmannMachine"],
    }
)

dashes = plotting_config["dashes"]
dashes.update(
    {
        "DataReuploadingClassifierSeparable": "(3, 2)",
        "DressedQuantumCircuitClassifierSeparable": "(3, 2)",
        "QuantumBoltzmannMachineSeparable": "(3, 2)",
    }
)
dashes = {k: eval(v) for k, v in dashes.items()}
markers = plotting_config["marker"]
markers.update(
    {
        "DataReuploadingClassifierSeparable": "^",
        "DressedQuantumCircuitClassifierSeparable": "h",
        "QuantumBoltzmannMachineSeparable": "H",
    }
)

for clf_type, out_name in zip(
    clfs, ["drc-separable", "dqcc-separable", "qbm-separable"]
):

    for dataname, (
        path_to_datafolder,
        dataset_name0,
        dataset_name1,
        xlabel,
    ) in datasets.items():

        df = pd.DataFrame(columns=["n_features", "test_acc", "train_acc", "clf"])
        df = df.rename(columns={"clf": "Model"})

        for clf in clf_type:

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

        fig, ax = plt.subplots(1, 1, figsize=(3, 3))

        sns.lineplot(
            ax=ax,
            data=df,
            x="n",
            y="test_acc",
            hue="Model",
            palette=colors,
            markers=markers,
            dashes=dashes,
           # legend=True,
            legend=False,
            style="Model",
        )
        sns.despine()

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim((0.7, 1.05))
        ax.set_xlim((2, 9))
        ax.set_ylabel("accuracy")
        ax.set_xlabel(xlabel)
        ax.set_title("test")

        fig.suptitle(f"{dataname}", fontsize=15)
        ax.grid(axis="y")
        #ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.4))
        plt.tight_layout()
        plt.savefig(f"figures/score-{dataname}-{out_name}.svg", bbox_inches='tight')
        plt.show()
