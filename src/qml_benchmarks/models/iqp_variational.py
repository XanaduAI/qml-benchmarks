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


class IQPVariationalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        repeats=1,
        n_layers=10,
        learning_rate=0.001,
        batch_size=32,
        use_jax=False,
        jit=True,
        vmap=False,
        max_vmap=None,
        max_steps=10000,
        convergence_interval=200,
        random_state=42,
        scaling=1.0,
        dev_type="default.qubit",
        qnode_kwargs={},
    ):
        r"""
        Variational verison of the classifier from https://arxiv.org/pdf/1804.11326v2.pdf.
        The model is a standard variational quantum classifier

        .. math::

            f(x)=\langle 0 \vert U^\dagger(x)V^\dagger(\theta) H V(\theta)U(x)\vert 0 \rangle

        where the data embedding unitary :math:`U(x)` is based on an IQP circuit stucture and implemented via
        pennylane.IQPEmbedding, and the trainable unitay :math:`V(\theta)` is implemented via
        pennylane.StronglyEntanglingLayers.

        The model is trained using a linear loss function equivalent to the probability of incorrect classification.

        Args:
            repeats (int): Number of times to repeat the IQP embedding circuit structure.
            n_layers (int): Number of layers in the variational part of the circuit.
            learning_rate (float): Learning rate for gradient descent.
            batch_size (int): Size of batches used for computing paraemeter updates.
            use_jax (bool): Whether to use jax. If False, no jitting and vmapping is performed either.
            jit (bool): Whether to use just in time compilation.
            vmap (bool): Whether to use jax.vmap.
            max_vmap (int or None): The maximum size of a chunk to vectorise over. Lower values use less memory.
                must divide batch_size.
            convergence_interval (int): If loss does not change significantly in this interval, stop training.
            max_steps (int): Maximum number of training steps. A warning will be raised if training did not converge.
            dev_type (str): string specifying the pennylane device type; e.g. 'default.qubit'.
            qnode_kwargs (str): the keyword arguments passed to the circuit qnode.
            random_state (int): Seed used for pseudorandom number generation.
        """

        # attributes that do not depend on data
        self.repeats = repeats
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.convergence_interval = convergence_interval
        self.batch_size = batch_size
        self.dev_type = dev_type
        self.qnode_kwargs = qnode_kwargs
        self.use_jax = use_jax
        self.vmap = vmap
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
        if self.use_jax:
            return jax.random.PRNGKey(self.rng.integers(1000000))
        return self.rng.integers(1000000)

    def construct_model(self):
        dev = qml.device(self.dev_type, wires=self.n_qubits_)

        if self.use_jax:
            @qml.qnode(dev, **self.qnode_kwargs)
            def circuit(params, x):
                """
                The variational circuit from the plots. Uses an IQP data embedding.
                We use the same observable as in the plots.
                """
                qml.IQPEmbedding(x, wires=range(self.n_qubits_), n_repeats=self.repeats)
                qml.StronglyEntanglingLayers(
                    params["weights"], wires=range(self.n_qubits_), imprimitive=qml.CZ
                )
                return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        else:
            @qml.qnode(dev, **self.qnode_kwargs)
            def circuit(weights, x):
                """
                The variational circuit from the plots. Uses an IQP data embedding.
                We use the same observable as in the plots.
                """
                qml.IQPEmbedding(x, wires=range(self.n_qubits_), n_repeats=self.repeats)
                qml.StronglyEntanglingLayers(
                    weights, wires=range(self.n_qubits_), imprimitive=qml.CZ
                )
                return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        self.circuit = circuit

        if self.use_jax and self.jit:
            circuit = jax.jit(circuit)

        if self.use_jax:
            if self.vmap:
                # use jax and batch feed the circuit
                self.forward = jax.vmap(circuit, in_axes=(None, 0))
                self.chunked_forward = chunk_vmapped_fn(self.forward, 1, self.max_vmap)
            else:
                # use jax but do not batch feed the circuit
                def forward(params, X):
                    return jnp.stack([circuit(params, x) for x in X])

                self.forward = forward
        else:
            # use autograd and do not batch feed the circuit
            def forward(params, X):
                return pnp.array([circuit(params, x) for x in X])

            self.forward = forward

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

        self.n_qubits_ = n_features
        self.initialize_params()
        self.construct_model()

    def initialize_params(self):
        # initialise the trainable parameters

        if self.use_jax:
            weights = (
                2
                * jnp.pi
                * jax.random.uniform(
                    shape=(self.n_layers, self.n_qubits_, 3), key=self.generate_key()
                )
            )
        else:
            weights = (
                2
                * np.pi
                * np.random.uniform(
                    size=(self.n_layers, self.n_qubits_, 3)
                )
            )
            weights = pnp.array(weights, requires_grad=True)

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

        if self.use_jax:

            optimizer = optax.adam

            def loss_fn(params, X, y):
                expvals = self.forward(params, X)
                probs = (1 - expvals * y) / 2  # the probs of incorrect classification
                return jnp.mean(probs)

            if self.jit:
                loss_fn = jax.jit(loss_fn)

            self.params_ = train(
                self,
                loss_fn,
                optimizer,
                X,
                y,
                self.generate_key,
                convergence_interval=self.convergence_interval,
            )

        else:
            X = pnp.array(X, requires_grad=False)
            y = pnp.array(y, requires_grad=False)
            optimizer = qml.AdamOptimizer

            def loss_fn(weights, X, y):
                expvals = self.forward(weights, X)
                probs = (1 - expvals * y) / 2  # the probs of incorrect classification
                return pnp.mean(probs)

            self.params_ = train_without_jax(self, loss_fn, optimizer, X, y, self.generate_key)

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
        if self.use_jax:
            if self.vmap:
                predictions = self.chunked_forward(self.params_, X)
            else:
                predictions = self.forward(self.params_, X)
        else:
            predictions = self.forward(self.params_["weights"], X)
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
