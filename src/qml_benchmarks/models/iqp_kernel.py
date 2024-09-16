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


class IQPKernelClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        svm=SVC(kernel="precomputed", probability=True),
        repeats=2,
        C=1.0,
        jit=False,
        random_state=42,
        scaling=1.0,
        max_vmap=250,
        dev_type="default.qubit",
        qnode_kwargs={"interface": "jax-jit", "diff_method": None},
    ):
        r"""
        Kernel version of the classifier from https://arxiv.org/pdf/1804.11326v2.pdf.
        The kernel function is given by

        .. math::
            k(x,x')=\vert\langle 0 \vert U^\dagger(x')U(x)\vert 0 \rangle\vert^2

        where :math:`U(x)` is an IQP circuit implemented via Pennylane's `IQPEmbedding`.

        We precompute the kernel matrix from the data directly, and pass it to scikit-learn's support vector
        classifier SVC. This  allows us to benefit from JAX parallelisation when computing the kernel
        matrices.

        The model requires evaluating a number of circuits given by the square of the number of data
        samples, and is therefore only appropriate for relatively small datasets.

        Args:
            svm (sklearn.svm.SVC): scikit-learn SVM class object used to fit the model from the kernel matrix
            repeats (int): number of times the IQP structure is repeated in the embedding circuit.
            C (float): regularization parameter for SVC. Lower values imply stronger regularization.
            jit (bool): Whether to use just in time compilation.
            dev_type (str): string specifying the pennylane device type; e.g. 'default.qubit'.
            max_vmap (int or None): The maximum size of a chunk to vectorise over. Lower values use less memory.
                must divide batch_size.
            qnode_kwargs (str): the key word arguments passed to the circuit qnode.
            scaling (float): Factor by which to scale the input data.
            random_state (int): seed used for reproducibility.
        """
        # attributes that do not depend on data
        self.repeats = repeats
        self.C = C
        self.jit = jit
        self.max_vmap = max_vmap
        self.svm = svm
        self.dev_type = dev_type
        self.qnode_kwargs = qnode_kwargs
        self.scaling = scaling
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        # data-dependant attributes
        # which will be initialised by calling "fit"
        self.params_ = None
        self.n_qubits_ = None
        self.scaler = None  # data scaler will be fitted on training data
        self.circuit = None

    def generate_key(self):
        return jax.random.PRNGKey(self.rng.integers(1000000))

    def construct_circuit(self):
        dev = qml.device(self.dev_type, wires=self.n_qubits_)

        @qml.qnode(dev, **self.qnode_kwargs)
        def circuit(x):
            """
            circuit used to precomute the kernel matrix K(x_1,x_2).
            Args:
                x (np.array): vector of length 2*num_feature that is the concatenation of x_1 and x_2

            Returns:
                (float) the value of the kernel fucntion K(x_1,x_2)
            """
            qml.IQPEmbedding(
                x[: self.n_qubits_], wires=range(self.n_qubits_), n_repeats=self.repeats
            )
            qml.adjoint(
                qml.IQPEmbedding(
                    x[self.n_qubits_ :],
                    wires=range(self.n_qubits_),
                    n_repeats=self.repeats,
                )
            )
            return qml.probs()

        self.circuit = circuit

        if self.jit:
            circuit = jax.jit(circuit)
        return circuit

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

        # concatenate all pairs of vectors
        Z = jnp.array(
            [np.concatenate((X1[i], X2[j])) for i in range(dim1) for j in range(dim2)]
        )

        circuit = self.construct_circuit()
        self.batched_circuit = chunk_vmapped_fn(
            jax.vmap(circuit, 0), start=0, max_vmap=self.max_vmap
        )
        kernel_values = self.batched_circuit(Z)[:, 0]

        # reshape the values into the kernel matrix
        kernel_matrix = np.reshape(kernel_values, (dim1, dim2))

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

        self.n_qubits_ = n_features

        self.construct_circuit()

    def fit(self, X, y):
        """Fit the model to data X and labels y. Uses sklearn's SVM classifier

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)
            y (np.ndarray): Labels of shape (n_samples,)
        """

        self.svm.random_state = self.rng.integers(100000)

        self.initialize(X.shape[1], np.unique(y))

        self.scaler = MinMaxScaler(feature_range=(-np.pi / 2, np.pi / 2))
        self.scaler.fit(X)
        X = self.transform(X)

        self.params_ = {"x_train": X}
        kernel_matrix = self.precompute_kernel(X, X)

        start = time.time()
        # we are updating this value here, in case it was
        # changed after initialising the model
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
        kernel_matrix = self.precompute_kernel(X, self.params_["x_train"])
        return self.svm.predict(kernel_matrix)

    def predict_proba(self, X):
        """Predict label probabilities for data X.
        note that this may be inconsistent with predict; see the sklearn docummentation for details.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)

        Returns:
            y_pred_proba (np.ndarray): Predicted label probabilities of shape
            (n_samples, n_classes)
        """

        if "x_train" not in self.params_:
            raise ValueError("Model cannot predict without fitting to data first.")

        X = self.transform(X)
        kernel_matrix = self.precompute_kernel(X, self.params_["x_train"])
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
