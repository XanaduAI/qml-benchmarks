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
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler
from qml_benchmarks.model_utils import *

jax.config.update("jax_enable_x64", True)


class VanillaQNN(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        embedding_layers=2,
        variational_layers=3,
        learning_rate=0.01,
        batch_size=32,
        max_vmap=None,
        jit=True,
        max_steps=10000,
        convergence_threshold=1e-6,
        random_state=42,
        scaling=1.0,
        dev_type="default.qubit.jax",
        qnode_kwargs={"interface": "jax"},
    ):
        """
        A vanilla implementation of a quantum neural network with layer-wise angle embedding and a layered
        variational circuit.

        Args:
            embedding_layers (int): number of times to repeat the embedding circuit structure.
            variational_layers (int): number of layers in the variational part of the circuit.
            learning_rate (float): learning rate for gradient descent.
            batch_size (int): Number of data points to subsample.
            max_vmap (int): The largest size of vmap used (to control memory)
            jit (bool): Whether to use just in time compilation.
            convergence_threshold (float): If loss changes less than this threshold for 10 consecutive steps we stop training.
            max_steps (int): Maximum number of training steps. A warning will be raised if training did not converge.
            dev_type (str): string specifying the pennylane device type; e.g. 'default.qubit'.
            qnode_kwargs (str): the keyword arguments passed to the circuit qnode.
            random_state (int): Seed used for pseudorandom number generation.
        """

        # attributes that do not depend on data
        self.embedding_layers = embedding_layers
        self.variational_layers = variational_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.convergence_threshold = convergence_threshold
        self.batch_size = batch_size
        self.dev_type = dev_type
        self.qnode_kwargs = qnode_kwargs
        self.jit = jit
        self.scaling = scaling
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        if max_vmap is None:
            self.max_vmap = self.batch_size
        else:
            self.max_vmap = max_vmap

        # data-dependant attributes
        # which will be initialised by calling "fit"
        self.params_ = None  # Dictionary containing the trainable parameters
        self.n_qubits_ = None
        self.scaler = None  # data scaler will be fitted on training data
        self.circuit = None

    def generate_key(self):
        return jax.random.PRNGKey(self.rng.integers(1000000))

    def construct_model(self):
        dev = qml.device(self.dev_type, wires=self.n_qubits_)

        @qml.qnode(dev, **self.qnode_kwargs)
        def circuit(params, x):
            """
            The variational circuit from the plots. Uses an IQP data embedding.
            We use the same observable as in the plots.
            """
            for i in range(self.embedding_layers):
                for j in range(self.n_qubits_):
                    qml.RX(x[j], wires=j)
                for i in range(self.n_qubits_):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits_])

            qml.StronglyEntanglingLayers(
                params["weights"], wires=range(self.n_qubits_), imprimitive=qml.CZ
            )
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

        if self.jit:
            circuit = jax.jit(circuit)
        self.forward = jax.vmap(circuit, in_axes=(None, 0))
        self.chunked_forward = chunk_vmapped_fn(self.forward, 1, self.max_vmap)

        return self.forward

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

        self.n_features = n_features
        self.n_qubits_ = n_features
        self.initialize_params()
        self.construct_model()

    def initialize_params(self):
        # initialise the trainable parameters
        weights = (
            2
            * jnp.pi
            * jax.random.uniform(
                shape=(self.variational_layers, self.n_qubits_, 3),
                key=self.generate_key(),
            )
        )
        self.params_ = {"weights": weights}

    def fit(self, X, y):
        """Fit the model to data X and labels y.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)
            y (np.ndarray): Labels of shape (n_samples,)
        """

        self.initialize(n_features=X.shape[1], classes=np.unique(y))

        self.scaler = MinMaxScaler(feature_range=(-np.pi / 2, np.pi / 2))
        self.scaler.fit(X)
        X = self.transform(X)

        optimizer = optax.adam

        def loss_fn(params, X, y):
            # we multiply by 6 because a relevant domain of the sigmoid function is [-6,6]
            vals = self.forward(params, X) * 6
            y = jax.nn.relu(y)  # convert to 0,1
            return jnp.mean(optax.sigmoid_binary_cross_entropy(vals, y))

        if self.jit:
            loss_fn = jax.jit(loss_fn)
        self.params_ = train(self, loss_fn, optimizer, X, y, self.generate_key)

        return self

    def predict(self, X):
        """Predict labels for data X.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)

        Returns:
            y_pred (np.ndarray): Predicted labels of shape (n_samples,)
        """
        predictions = self.predict_proba(X)
        mapped_predictions = np.argmax(predictions, axis=1)
        return np.take(self.classes_, mapped_predictions)

    def predict_proba(self, X):
        """Predict label probabilities for data X.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)

        Returns:
            y_pred_proba (np.ndarray): Predicted label probabilities of shape
            (n_samples, n_classes)
        """
        X = self.transform(X)
        predictions = self.chunked_forward(self.params_, X)
        predictions_2d = np.c_[(1 - predictions) / 2, (1 + predictions) / 2]
        return predictions_2d

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
