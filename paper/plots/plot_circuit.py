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

import pennylane as qml
from jax import numpy as jnp
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)
from qml_benchmarks.models import *

models_to_plot = [
    CircuitCentricClassifier(),
    DataReuploadingClassifier(),
    DressedQuantumCircuitClassifier(),
    IQPKernelClassifier(),
    IQPVariationalClassifier(),
    ProjectedQuantumKernel(trotter_steps=1, t=1 / 4),
    QuantumKitchenSinks(n_episodes=1),
    QuantumMetricLearner(),
    QuanvolutionalNeuralNetwork(threshold=0.2),
    SeparableVariationalClassifier(),
    SeparableKernelClassifier(encoding_layers=2),
    TreeTensorClassifier(),
    VanillaQNN(),
    WeiNet(),
]

for clf in models_to_plot:

    clf_name = clf.__class__.__name__

    if clf_name == "WeiNet":
        clf.initialize(16)
        x = jnp.array([[0.1 * i for i in range(1, 17)]])
    else:
        clf.initialize(4)
        x = jnp.array([[0.1, 0.2, 0.3, 0.4]])

    if clf_name == "QuantumKitchenSinks":
        # Note: QuantumKitchenSinks.transform already calls the circuit
        x = x[0]
    elif clf_name == "QuanvolutionalNeuralNetwork":
        # don't transform because
        x = jnp.heaviside(x - 0.2, 0.0)[0]
    else:
        x = clf.transform(x, preprocess=False)[0]

    if clf_name in ["IQPKernelClassifier", "ProjectedQuantumKernel"]:
        # kernel circuits take two inputs that are concatenated, i.e. x = [x, x']
        x = jnp.concatenate([x, x])

    if clf_name in [
        "IQPKernelClassifier",
        "ProjectedQuantumKernel",
        "QuantumKitchenSinks",
        "QuanvolutionalNeuralNetwork",
        "WeiNet",
    ]:
        fig, ax = qml.draw_mpl(
            clf.circuit, style="pennylane_sketch", fontsize="xx-large", decimals=1
        )(x)
    elif clf_name == "QuantumMetricLearner":
        fig, ax = qml.draw_mpl(
            clf.circuit, style="pennylane_sketch", fontsize="xx-large", decimals=1
        )(clf.params_, x, x)
    elif clf_name == "SeparableVariationalClassifier":
        fig, ax = qml.draw_mpl(
            clf.circuit, style="pennylane_sketch", fontsize="xx-large", decimals=1
        )(clf.params_["weights"][0], x[0])
    elif clf_name == "SeparableKernelClassifier":
        fig, ax = qml.draw_mpl(
            clf.circuit, style="pennylane_sketch", fontsize="xx-large", decimals=1
        )(jnp.stack([x[0], x[0]]))
    else:
        fig, ax = qml.draw_mpl(
            clf.circuit, style="pennylane_sketch", fontsize="xx-large", decimals=1
        )(clf.params_, x)

    plt.savefig(f"figures/circuit-{clf.__class__.__name__}.svg")
    plt.show()
