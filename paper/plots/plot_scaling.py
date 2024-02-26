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
Script to reproduce Fig. 18 in the plots, showing the results of rescaling amplitude encoded data.
"""
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import os
import yaml

os.makedirs("figures", exist_ok=True)


sns.set(rc={"figure.figsize": (6, 4)})
sns.set(font_scale=1.2)
sns.set_style("white")

data_qnn = pandas.read_csv("scaling_results/scaling-results_qnn.csv")
data_kernel = pandas.read_csv("scaling_results/scaling-results_kernel.csv")

datasets = [
    "mnist_3-5_6d",
    "linearly_separable_6d",
    "hidden_manifold-6manifold-6d",
    "two_manifold-5degree-0.1offset-6d",
]
dataset_names = ["MNIST PCA", "LINEARLY SEPARABLE", "HIDDEN MANIFOLD", "TWO MANIFOLD"]

models = ["CircuitCentricClassifier", "TreeTensorClassifier"]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)

for ax, dataset, dataset_name in zip([ax1, ax2, ax3, ax4], datasets, dataset_names):

    data = pandas.concat([data_qnn, data_kernel], axis=0, ignore_index=True)
    data = data[data["dataset"] == dataset]
    data = data[data["model"].isin(models)]

    with open("plotting_standards.yaml", "r") as stream:
        plotting_config = yaml.safe_load(stream)
    colors = plotting_config["color"]
    dashes = plotting_config["dashes"]
    markers = plotting_config["marker"]

    ax.axvline(x=1.0, color='gray', linestyle='--')

    sns.lineplot(
        ax=ax,
        data=data,
        x="scaling",
        y="score",
        style="model",
        hue="model",
        palette=colors,
        markers=markers,
    )
    sns.despine()
    ax.set_xlabel("data scaling")
    ax.set_ylabel("accuracy")
    ax.set_xscale("log")
    ax.set_title(dataset_name)
    ax.legend().set_visible(False)

handles, labels = plt.gca().get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, loc='lower center', bbox_to_anchor=(0.5, -0.15))

plt.tight_layout()
plt.savefig(f"figures/scaling.png", bbox_inches="tight")
plt.show()
