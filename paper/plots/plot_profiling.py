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
Script to reproduce Fig. 6 in the plots, showing the average difficulty score of selected benchmarks
according to the ECol methodology.
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)
import seaborn as sns

sns.set(rc={"figure.figsize": (6, 3)})
sns.set(font_scale=1.1)
sns.set_style("white")

datasets = {
    "mnist_pca": ["mnist_3-5_", "d-250", "MNIST PCA-"],
    "linearly-separable": ["linearly_separable_", "d", "LINEARLY SEPARABLE"],
    "hmm": ["hidden_manifold-6manifold-", "d", "HIDDEN MANIFOLD"],
    "two-manifold": ["two_curves-5degree-0.1offset-", "d", "TWO CURVES"],
    "hmm-diff": ["hidden_manifold-10d-", "manifold", "HIDDEN MANIFOLD DIFF"],
    "two-manifold-diff": ["two_curves-10d-", "degree", "TWO CURVES DIFF"],
    "hyperplanes-diff": ["hyperplanes-10d-from3d-", "n", "HYPERPLANES DIFF"],
}

df = pd.DataFrame(columns=["measure", "dataset"] + [f"score_{i}" for i in range(2, 21)])

for dataset, (dataset_name0, dataset_name1, display_name) in datasets.items():

    # gather results of profiling
    df_data = pd.DataFrame(columns=["measure", "score"])
    for k in range(2, 21):
        new_df = pd.read_csv(
            f"profiling_results/PROFILE-{dataset_name0}{k}{dataset_name1}_train.csv",
            header=0,
            names=["measure", "score"],
        )
        df_data = df_data.merge(
            new_df, how="outer", on=["measure"], suffixes=("", f"_{k}")
        )
    df_data["dataset"] = [display_name] * 22
    df = pd.concat([df, df_data], axis=0)

# drop unnecessary information
df = df[~df["measure"].str.contains(".sd")]
df = df.drop(["score", "measure"], axis=1)

# average over datasets
df = df.groupby("dataset", axis=0).mean()
df = df.transpose()
df = df.rename(index={f"score_{i}": str(i) for i in range(2, 21)})
df = df[
    [
        "LINEARLY SEPARABLE",
        "MNIST PCA-",
        "HIDDEN MANIFOLD",
        "TWO CURVES",
        "HYPERPLANES DIFF",
        "HIDDEN MANIFOLD DIFF",
        "TWO CURVES DIFF",
    ]
]

df.plot(
    style={
        "LINEARLY SEPARABLE": "gold",
        "MNIST PCA-": "darkorange",
        "HIDDEN MANIFOLD": "darkgoldenrod",
        "TWO CURVES": "tan",
        "HIDDEN MANIFOLD DIFF": "darkgreen",
        "TWO CURVES DIFF": "red",
        "HYPERPLANES DIFF": "darkblue",
    }
)
sns.despine()

plt.xlabel("variable")
plt.ylabel("difficulty")
plt.legend(frameon=False, bbox_to_anchor=(1, 1), loc="upper left")
plt.tight_layout()
plt.savefig("figures/profiling.png")

plt.show()
