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

"""Hyperparameter settings for all models"""

hyper_parameter_settings = {
    "CircuitCentricClassifier": {
        "max_vmap": {"type": "list", "dtype": "int", "val": [1]},
        "batch_size": {"type": "list", "dtype": "int", "val": [32]},
        "learning_rate": {"type": "list", "dtype": "float", "val": [0.001, 0.01, 0.1]},
        "n_input_copies": {"type": "list", "dtype": "int", "val": [1, 2, 3]},
        "n_layers": {"type": "list", "dtype": "int", "val": [1, 5, 10]},
    },
    "ClassicalFourierMPS": {
        "batch_size": {"type": "list", "dtype": "int", "val": [32]},
        "learning_rate": {"type": "list", "dtype": "float", "val": [0.01]},
        "regularisation": {"type": "list", "dtype": "float", "val": [0.1, 0.01]},
        "degree": {"type": "list", "dtype": "int", "val": [1, 5, 10]},
        "bond_dimension": {"type": "list", "dtype": "int", "val": [2, 5, 10]},
    },
    "DataReuploadingClassifier": {
        "max_vmap": {"type": "list", "dtype": "int", "val": [1]},
        "batch_size": {"type": "list", "dtype": "int", "val": [32]},
        "learning_rate": {"type": "list", "dtype": "float", "val": [0.001, 0.01, 0.1]},
        "n_layers": {"type": "list", "dtype": "int", "val": [1, 5, 10, 15]},
        "observable_type": {
            "type": "list",
            "dtype": "str",
            "val": ["single", "half", "full"],
        },
    },
    "DressedQuantumCircuitClassifier": {
        "max_vmap": {"type": "list", "dtype": "int", "val": [1]},
        "batch_size": {"type": "list", "dtype": "int", "val": [32]},
        "learning_rate": {"type": "list", "dtype": "float", "val": [0.001, 0.01, 0.1]},
        "n_layers": {"type": "list", "dtype": "int", "val": [1, 5, 10, 15]},
    },
    "IQPKernelClassifier": {
        "max_vmap": {"type": "list", "dtype": "int", "val": [1]},
        "repeats": {"type": "list", "dtype": "int", "val": [1, 5, 10]},
        "C": {"type": "list", "dtype": "float", "val": [0.1, 1, 10, 100]},
    },
    "IQPVariationalClassifier": {
        "max_vmap": {"type": "list", "dtype": "int", "val": [1]},
        "batch_size": {"type": "list", "dtype": "int", "val": [32]},
        "learning_rate": {"type": "list", "dtype": "float", "val": [0.001, 0.01, 0.1]},
        "repeats": {"type": "list", "dtype": "int", "val": [1, 5, 10]},
        "n_layers": {"type": "list", "dtype": "int", "val": [1, 5, 10, 15]},
    },
    "ProjectedQuantumKernel": {
        "max_vmap": {"type": "list", "dtype": "int", "val": [100]},
        "gamma_factor": {"type": "list", "dtype": "float", "val": [0.1, 1, 10]},
        "C": {"type": "list", "dtype": "float", "val": [0.1, 1, 10, 100]},
        "trotter_steps": {"type": "list", "dtype": "int", "val": [1, 3, 5]},
        "t": {"type": "list", "dtype": "float", "val": [0.01, 0.1, 1.0]},
    },
    "QuantumKitchenSinks": {
        "max_vmap": {"type": "list", "dtype": "int", "val": [1]},
        "n_qfeatures": {"type": "list", "dtype": "str", "val": ["full", "half"]},
        "n_episodes": {"type": "list", "dtype": "int", "val": [10, 100, 500, 2000]},
    },
    "QuantumMetricLearner": {
        "max_vmap": {"type": "list", "dtype": "int", "val": [1]},
        "batch_size": {"type": "list", "dtype": "int", "val": [16]},
        "learning_rate": {"type": "list", "dtype": "float", "val": [0.001, 0.01, 0.1]},
        "n_layers": {"type": "list", "dtype": "int", "val": [1, 3, 4]},
    },
    "QuantumBoltzmannMachine": {
        "max_vmap": {"type": "list", "dtype": "int", "val": [32]},
        "batch_size": {"type": "list", "dtype": "int", "val": [32]},
        "learning_rate": {"type": "list", "dtype": "float", "val": [0.001, 0.01, 0.1]},
        "visible_qubits": {
            "type": "list",
            "dtype": "str",
            "val": ["single", "half", "full"],
        },
        "temperature": {"type": "list", "dtype": "float", "val": [1, 10, 100]},
    },
    "TreeTensorClassifier": {
        "max_vmap": {"type": "list", "dtype": "int", "val": [1]},
        "batch_size": {"type": "list", "dtype": "int", "val": [32]},
        "learning_rate": {"type": "list", "dtype": "float", "val": [0.001, 0.01, 0.1]},
    },
    "ParallelGradients": {
        "max_vmap": {"type": "list", "dtype": "int", "val": [1]},
        "batch_size": {"type": "list", "dtype": "int", "val": [32]},
        "learning_rate": {"type": "list", "dtype": "float", "val": [0.001, 0.01, 0.1]},
        "encoding_layers": {"type": "list", "dtype": "int", "val": [1, 3, 5]},
        "degree": {"type": "list", "dtype": "int", "val": [2, 3, 4]},
    },
    "QuanvolutionalNeuralNetwork": {
        "max_vmap": {"type": "list", "dtype": "int", "val": [32]},
        "batch_size": {"type": "list", "dtype": "int", "val": [32]},
        "learning_rate": {
            "type": "list",
            "dtype": "float",
            "val": [0.0001, 0.001, 0.01],
        },
        "n_qchannels": {"type": "list", "dtype": "int", "val": [1, 5, 10]},
        "qkernel_shape": {"type": "list", "dtype": "int", "val": [2, 3]},
        "kernel_shape": {"type": "list", "dtype": "int", "val": [2, 3, 5]},
    },
    "WeiNet": {
        "max_vmap": {"type": "list", "dtype": "int", "val": [1]},
        "batch_size": {"type": "list", "dtype": "int", "val": [32]},
        "learning_rate": {
            "type": "list",
            "dtype": "float",
            "val": [0.0001, 0.001, 0.01],
        },
        "filter_name": {
            "type": "list",
            "dtype": "str",
            "val": ["edge_detect", "smooth", "sharpen"],
        },
    },
    "SeparableVariationalClassifier": {
        "batch_size": {"type": "list", "dtype": "int", "val": [32]},
        "learning_rate": {"type": "list", "dtype": "float", "val": [0.001, 0.01, 0.1]},
        "encoding_layers": {"type": "list", "dtype": "int", "val": [1, 3, 5, 10]},
    },
    "SeparableKernelClassifier": {
        "C": {"type": "list", "dtype": "float", "val": [0.1, 1, 10, 100]},
        "encoding_layers": {"type": "list", "dtype": "int", "val": [1, 3, 5, 10]},
    },
    "ConvolutionalNeuralNetwork": {
        "batch_size": {"type": "list", "dtype": "int", "val": [32]},
        "learning_rate": {
            "type": "list",
            "dtype": "float",
            "val": [0.0001, 0.001, 0.01],
        },
        "kernel_shape": {"type": "list", "dtype": "int", "val": [2, 3, 5]},
    },
    "SVC": {
        "gamma": {"type": "list", "dtype": "float", "val": [0.001, 0.01, 0.1, 1]},
        "C": {"type": "list", "dtype": "float", "val": [0.1, 1, 10, 100]},
    },
    "SVClinear": {"C": {"type": "list", "dtype": "float", "val": [0.1, 1, 10, 100]}},
    "MLPClassifier": {
        "batch_size": {"type": "list", "dtype": "int", "val": [32]},
        "learning_rate_init": {
            "type": "list",
            "dtype": "float",
            "val": [0.001, 0.01, 0.1],
        },
        "hidden_layer_sizes": {
            "type": "list",
            "dtype": "tuple",
            "val": ["(100,)", "(10, 10, 10, 10)", "(50, 10, 5)"],
        },
        "alpha": {"type": "list", "dtype": "float", "val": [0.01, 0.001, 0.0001]},
    },
    "Perceptron": {"eta0": {"type": "list", "dtype": "float", "val": [0.1, 1, 10]}},
}
