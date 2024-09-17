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

import time
import pennylane as qml
import numpy as np
import jax
import jax.numpy as jnp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from qml_benchmarks.model_utils import chunk_vmapped_fn

jax.config.update("jax_enable_x64", True)


class ProjectedQuantumKernel(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        svm=SVC(kernel="precomputed", probability=True),
        gamma_factor=1.0,
        C=1.0,
        embedding="Hamiltonian",
        t=1.0 / 3,
        trotter_steps=5,
        jit=True,
        max_vmap=None,
        scaling=1.0,
        dev_type="default.qubit",
        qnode_kwargs={"interface": "jax-jit", "diff_method": None},
        random_state=42,
    ):
        r"""
        Kernel based classifier from https://arxiv.org/pdf/2011.01938v2.pdf.

        The Kernel function is

        .. math::
            k(x_i, x_j) = \exp (-\gamma  \sum_k \sum_{P \in \{X, Y, Z\}} ( \text{tr}(P \rho(x_i)_k)
            - \text{tr}( P \rho(x_j)_k))^2)

        where :math:`\rho_k(x_i)` is the reduced state of the kth qubit of the data embedded density matrix and
        :math:`\gamma` is a hyperparameter of the model that we scale from the default value given in the plots.

        For embedding='Hamiltonian' a layer or random single qubit rotations are performed, followed by a Hamiltonian
        time evolution corresponding to a trotterised evolution of a Heisenberg Hamiltonian:

        .. math::
            \prod_{j=1}^n \exp(-i \frac{t}{L} x_{j} (X_j X_{j+1} + Y_j Y_{j+1} + Z_j Z_{j+1})).

        where :math:`t` and :math:`L` are the evolution time and the number of trotter steps that are controlled by
        hyperparameters `t` and `trotter_steps`.

        For emedding='IQP' an IQP embedding is used via PennyLanes's IQPEmbedding class.

        We precompute the kernel matrix from data and pass it to scikit-learn's support vector machine class SVC,
        which fits a classifier.

        Args:
            svm (sklearn.svm.SVC): scikit-learn SVC class object.
            gamma_factor (float): the factor that multiplies the default scaling parameter in the kernel.
            C (float): regularization parameter when fitting the kernel model.
            embedding (str): The choice of embedding circuit used to construct the kernel.
                Either 'IQP' or 'Hamiltonian'.
            t (float): The evolution time used in the 'Hamiltonian' data embedding. The time is
                given by n_features*t.
            trotter_steps (int): the number of trotter steps used in the 'Hamiltonian' embedding circuit.
            max_vmap (int or None): The maximum size of a chunk to vectorise over. Lower values use less memory.
                must divide batch_size.
            jit (bool): Whether to use just in time compilation.
            dev_type (str): string specifying the pennylane device type; e.g. 'default.qubit'
            qnode_kwargs (str): the key word arguments passed to the circuit qnode.
            scaling (float): Factor by which to scale the input data.
            random_state (int): Seed used for pseudorandom number generation.
        """
        # attributes that do not depend on data
        self.gamma_factor = gamma_factor
        self.svm = svm
        self.C = C
        self.embedding = embedding
        self.t = t
        self.trotter_steps = trotter_steps
        self.jit = jit
        self.dev_type = dev_type
        self.qnode_kwargs = qnode_kwargs
        self.scaling = scaling
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        if max_vmap is None:
            self.max_vmap = 50
        else:
            self.max_vmap = max_vmap

        # data-dependant attributes
        # which will be initialised by calling "fit"
        self.params_ = None
        self.n_qubits_ = None
        self.n_features_ = None
        self.rotation_angles_ = None  # for hamiltonian embedding
        self.scaler = None  # data scaler will be fitted on training data
        self.circuit = None

    def generate_key(self):
        return jax.random.PRNGKey(self.rng.integers(1000000))

    def construct_circuit(self):
        """
        Constructs the circuit to get the expvals of a given qubit and Pauli operator
        We will use JAX to parallelize over these circuits in precompute kernel.
        Args:
            P: a pennylane Pauli X,Y,Z operator on a given qubit
        """
        if self.embedding == "IQP":

            def embedding(x):
                qml.IQPEmbedding(x, wires=range(self.n_qubits_), n_repeats=2)

        elif self.embedding == "Hamiltonian":

            def embedding(x):
                evol_time = self.t / self.trotter_steps * (self.n_qubits_ - 1)
                for i in range(self.n_qubits_):
                    qml.Rot(
                        self.rotation_angles_[i, 0],
                        self.rotation_angles_[i, 1],
                        self.rotation_angles_[i, 2],
                        wires=i,
                    )
                for __ in range(self.trotter_steps):
                    for j in range(self.n_qubits_ - 1):
                        qml.IsingXX(x[j] * evol_time, wires=[j, j + 1])
                        qml.IsingYY(x[j] * evol_time, wires=[j, j + 1])
                        qml.IsingZZ(x[j] * evol_time, wires=[j, j + 1])

        dev = qml.device(self.dev_type, wires=self.n_qubits_)

        @qml.qnode(dev, **self.qnode_kwargs)
        def circuit(x):
            embedding(x)
            return (
                [qml.expval(qml.PauliX(wires=i)) for i in range(self.n_qubits_)]
                + [qml.expval(qml.PauliY(wires=i)) for i in range(self.n_qubits_)]
                + [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits_)]
            )

        self.circuit = circuit

        def circuit_as_array(x):
            return jnp.array(circuit(x))

        if self.jit:
            circuit_as_array = jax.jit(circuit_as_array)
        circuit_as_array = jax.vmap(circuit_as_array, in_axes=(0))
        circuit_as_array = chunk_vmapped_fn(circuit_as_array, 0, self.max_vmap)

        return circuit_as_array

    def precompute_kernel(self, X1, X2):
        """
        compute the kernel matrix relative to data sets X1 and X2
        Args:
            X1 (np.array): first dataset of input vectors
            X2 (np.array): second dataset of input vectors
        Returns:
            kernel_matrix (np.array): matrix of size (len(X1),len(X2)) with elements K(x_1,x_2)
        """
        dim1 = len(X1)
        dim2 = len(X2)

        # get all of the Pauli expvals needed to constrcut the kernel
        self.circuit = self.construct_circuit()

        valsX1 = np.array(self.circuit(X1))
        valsX1 = np.reshape(valsX1, (dim1, 3, -1))
        valsX2 = np.array(self.circuit(X2))
        valsX2 = np.reshape(valsX2, (dim2, 3, -1))

        valsX_X1 = valsX1[:, 0]
        valsX_X2 = valsX2[:, 0]
        valsY_X1 = valsX1[:, 1]
        valsY_X2 = valsX2[:, 1]
        valsZ_X1 = valsX1[:, 2]
        valsZ_X2 = valsX2[:, 2]

        all_vals_X1 = np.reshape(np.concatenate((valsX_X1, valsY_X1, valsZ_X1)), -1)
        default_gamma = 1 / np.var(all_vals_X1) / self.n_features_

        # construct kernel following plots
        kernel_matrix = np.zeros([dim1, dim2])

        for i in range(dim1):
            for j in range(dim2):
                sumX = sum(
                    [
                        (valsX_X1[i, q] - valsX_X2[j, q]) ** 2
                        for q in range(self.n_qubits_)
                    ]
                )
                sumY = sum(
                    [
                        (valsY_X1[i, q] - valsY_X2[j, q]) ** 2
                        for q in range(self.n_qubits_)
                    ]
                )
                sumZ = sum(
                    [
                        (valsZ_X1[i, q] - valsZ_X2[j, q]) ** 2
                        for q in range(self.n_qubits_)
                    ]
                )

                kernel_matrix[i, j] = np.exp(
                    -default_gamma * self.gamma_factor * (sumX + sumY + sumZ)
                )
        return kernel_matrix

    def initialize(self, n_features, classes=None):
        """Initialize attributes that depend on the number of features and the class labels.

        Args:
            n_features (int): Number of features that the classifier expects
            classes (array-like): class labels that the classifier expects
        """
        if classes is None:
            classes = [-1, 1]

        self.classes_ = classes
        self.n_classes_ = len(self.classes_)
        assert self.n_classes_ == 2
        assert 1 in self.classes_ and -1 in self.classes_

        self.n_features_ = n_features

        if self.embedding == "IQP":
            self.n_qubits_ = self.n_features_
        elif self.embedding == "Hamiltonian":
            self.n_qubits_ = self.n_features_ + 1
            self.rotation_angles_ = jnp.array(
                self.rng.uniform(size=(self.n_qubits_, 3)) * np.pi * 2
            )
        self.construct_circuit()

    def fit(self, X, y):
        """Fit the model to data X and labels y. Uses sklearn's SVM classifier

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)
            y (np.ndarray): Labels of shape (n_samples,)
        """
        self.svm.random_state = self.rng.integers(100000)

        self.initialize(X.shape[1], classes=np.unique(y))

        self.scaler = MinMaxScaler(feature_range=(-np.pi / 2, np.pi / 2))
        self.scaler.fit(X)
        X = self.transform(X)

        self.params_ = {"X_train": X}
        kernel_matrix = self.precompute_kernel(X, X)

        start = time.time()
        self.svm.C = self.C
        self.svm.fit(kernel_matrix, y)
        end = time.time()
        self.training_time_ = end - start

        return self

    def predict(self, X):
        """Predict labels for data X.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)

        Returns:
            y_pred (np.ndarray): Predicted labels of shape (n_samples,)
        """
        X = self.transform(X)
        kernel_matrix = self.precompute_kernel(X, self.params_["X_train"])
        return self.svm.predict(kernel_matrix)

    def predict_proba(self, X):
        """Predict label probabilities for data X.
        Note that this may be inconsistent with predict; see the sklearn docummentation for details.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)

        Returns:
            y_pred_proba (np.ndarray): Predicted label probabilities of shape
            (n_samples, n_classes)
        """

        if "X_train" not in self.params_:
            raise ValueError("Model cannot predict without fitting to data first.")

        X = self.transform(X)
        kernel_matrix = self.precompute_kernel(X, self.params_["X_train"])
        return self.svm.predict_proba(kernel_matrix)

    def transform(self, X, preprocess=True):
        """
        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)
        """
        if preprocess:
            if self.scaler is None:
                # if the model is unfitted, initialise the scaler here
                self.scaler = MinMaxScaler(feature_range=(-np.pi / 2, np.pi / 2))
                self.scaler.fit(X)
            X = self.scaler.transform(X)
        return X * self.scaling
