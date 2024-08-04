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

"""Module containing models to be used in benchmarks."""

from qml_benchmarks.models.circuit_centric import CircuitCentricClassifier
from qml_benchmarks.models.convolutional_neural_network import (
    ConvolutionalNeuralNetwork,
)
from qml_benchmarks.models.data_reuploading import (
    DataReuploadingClassifier,
    DataReuploadingClassifierNoScaling,
    DataReuploadingClassifierNoCost,
    DataReuploadingClassifierNoTrainableEmbedding,
    DataReuploadingClassifierSeparable,
)
from qml_benchmarks.models.dressed_quantum_circuit import (
    DressedQuantumCircuitClassifier,
    DressedQuantumCircuitClassifierOnlyNN,
    DressedQuantumCircuitClassifierSeparable,
)

from qml_benchmarks.models.iqp_kernel import IQPKernelClassifier
from qml_benchmarks.models.iqp_variational import IQPVariationalClassifier
from qml_benchmarks.models.projected_quantum_kernel import ProjectedQuantumKernel
from qml_benchmarks.models.quantum_boltzmann_machine import (
    QuantumBoltzmannMachine,
    QuantumBoltzmannMachineSeparable,
)
from qml_benchmarks.models.quantum_kitchen_sinks import QuantumKitchenSinks
from qml_benchmarks.models.quantum_metric_learning import QuantumMetricLearner
from qml_benchmarks.models.quanvolutional_neural_network import (
    QuanvolutionalNeuralNetwork,
)
from qml_benchmarks.models.separable import (
    SeparableVariationalClassifier,
    SeparableKernelClassifier,
)
from qml_benchmarks.models.tree_tensor import TreeTensorClassifier
from qml_benchmarks.models.vanilla_qnn import VanillaQNN
from qml_benchmarks.models.weinet import WeiNet

from sklearn.svm import SVC as SVC_base
from sklearn.neural_network import MLPClassifier as MLP

__all__ = [
    "CircuitCentricClassifier",
    "ConvolutionalNeuralNetwork",
    "DataReuploadingClassifier",
    "DataReuploadingClassifierNoScaling",
    "DataReuploadingClassifierNoCost",
    "DataReuploadingClassifierNoTrainableEmbedding",
    "DataReuploadingClassifierSeparable",
    "DressedQuantumCircuitClassifier",
    "DressedQuantumCircuitClassifierOnlyNN",
    "DressedQuantumCircuitClassifierSeparable",
    "IQPKernelClassifier",
    "IQPVariationalClassifier",
    "ProjectedQuantumKernel",
    "QuantumBoltzmannMachine",
    "QuantumBoltzmannMachineSeparable",
    "QuantumKitchenSinks",
    "QuantumMetricLearner",
    "QuanvolutionalNeuralNetwork",
    "SeparableVariationalClassifier",
    "SeparableKernelClassifier",
    "TreeTensorClassifier",
    "VanillaQNN",
    "WeiNet",
    "MLPClassifier",
    "SVC",
]


class MLPClassifier(MLP):
    def __init__(
        self,
        hidden_layer_sizes=(100, 100),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=200,
        shuffle=True,
        random_state=None,
        tol=1e-4,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10,
        max_fun=15000,
    ):
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
            max_fun=max_fun,
        )


class SVC(SVC_base):
    def __init__(
        self,
        C=1.0,
        degree=3,
        gamma="scale",
        coef0=0.0,
        shrinking=True,
        probability=False,
        tol=0.001,
        max_iter=-1,
        random_state=None,
    ):
        super().__init__(
            C=C,
            kernel="rbf",
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
        )
