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
Script to reproduce Fig. 3 in the plots, showing the effect of a positivity bias in a
small thought experiment simulation.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)
from scipy.stats import norm
import seaborn as sns

sns.set(rc={"figure.figsize": (8, 3)})
sns.set(font_scale=1.5)
sns.set_style("white")

palette = sns.color_palette('deep')

mean_quantum = 0.55
scale_quantum = 0.1
mean_class = 0.65
scale_class = 0.07

# conduct simulation
stats_bias = []
for researcher in range(100):
    experiment = []
    # sample a model performance from the classical model distribution
    classical_performance = np.random.normal(loc=mean_class, scale=scale_class)
    # sample 20 model performances from the quantum model distribution
    for design in range(20):
        quantum_performance = np.random.normal(loc=mean_quantum, scale=scale_quantum)
        experiment.append((quantum_performance, classical_performance))
    # select the experiment with the largest advantage of the quantum model
    experiment.sort(key=lambda res: res[1] - res[0])
    stats_bias.append(experiment[0])


# plot the results
x_axis = np.arange(0, 1, 0.001)
mx = max(
    np.max(norm.pdf(x_axis, mean_quantum, scale_quantum)),
    np.max(norm.pdf(x_axis, mean_class, scale_class)),
)

plt.fill_between(
    x_axis,
    norm.pdf(x_axis, mean_quantum, scale_quantum),
    color=palette[0],
    alpha=0.5,
    label="true quantum (pdf)",
)
bias_quantum = np.mean([s[0] for s in stats_bias])
plt.plot(
    [bias_quantum, bias_quantum],
    [0, mx],
    color=palette[0],
    linestyle="--",
    label="biased quantum (mean)",
)

plt.fill_between(
    x_axis,
    norm.pdf(x_axis, mean_class, scale_class),
    color=palette[1],
    alpha=0.5,
    label="true classical (pdf)",
)
bias_class = np.mean([s[1] for s in stats_bias])
plt.plot(
    [bias_class, bias_class],
    [0, mx],
    color=palette[1],
    linestyle="--",
    label="true classical (mean)",
)
sns.despine()

plt.ylabel("")
plt.xlabel("performance")
plt.xlim((0, 1))
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
plt.tight_layout()
plt.savefig("figures/positivity-bias.png")
plt.show()
