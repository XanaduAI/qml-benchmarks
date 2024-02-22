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
import jax
import jax.numpy as jnp
import optax
from sklearn.base import BaseEstimator, ClassifierMixin
from qml_benchmarks.model_utils import train
from qml_benchmarks.model_utils import chunk_vmapped_fn

jax.config.update("jax_enable_x64", True)


class TreeTensorClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        learning_rate=0.01,
        batch_size=32,
        max_steps=10000,
        convergence_interval=200,
        random_state=42,
        max_vmap=None,
        jit=True,
        scaling=1.0,
        dev_type="default.qubit.jax",
        qnode_kwargs={"interface": "jax-jit"},
    ):
        r"""
        Tree tensor network classifier from https://arxiv.org/abs/2011.06258v2 (see figure 1)

        This is a variational model where data is amplitude embedded and a trainiable circuit with a tree like
        structure is used for prediction. Due to the tree structure, the number of qubits must be a power of 2.

        In the plots, the data encoding state for a given input x is approximated  using a variational circuit.
        In practice, this means one has to train an encoding circuit for every training and test
        input. Since this is very expensive, we just use the exact amplitude encoded state. If the input data
        dimension is smaller than the state vector dimension, we pad with a value :math:`1/2^{n_qubits}`.

        The classification is performed via a Z measurement on the first qubit plus a trainable bias. Training
        is via the square loss.

        Args:
            learning_rate (float): Initial learning rate for gradient descent.
            batch_size (int): Size of batches used for computing parameter updates.
            convergence_interval (int): The number of loss values to consider to decide convergence.
            max_steps (int): Maximum number of training steps. A warning will be raised if training did not converge.
            random_state (int): Seed used for pseudorandom number generation
            max_vmap (int or None): The maximum size of a chunk to vectorise over. Lower values use less memory.
                must divide batch_size.
            jit (bool): Whether to use just in time compilation.
            scaling (float): Factor by which to scale the input data.
            dev_type (str): string specifying the pennylane device type; e.g. 'default.qubit'
            qnode_kwargs (str): the key word arguments passed to the circuit qnode
        """

        # attributes that do not depend on data
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.convergence_interval = convergence_interval
        self.batch_size = batch_size
        self.dev_type = dev_type
        self.jit = jit
        self.qnode_kwargs = qnode_kwargs
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
        self.circuit = None

    def generate_key(self):
        return jax.random.PRNGKey(self.rng.integers(1000000))

    def construct_model(self):
        dev = qml.device(self.dev_type, wires=self.n_qubits)
        n_layers = int(jnp.log2(self.n_qubits))

        @qml.qnode(dev, **self.qnode_kwargs)
        def circuit(params, x):
            """
            The tree tensor QNN from the plots
            """
            qml.AmplitudeEmbedding(x, normalize=True, wires=range(self.n_qubits))
            count = 0
            for layer in range(n_layers):
                for q in range(0, self.n_qubits, 2**layer):
                    qml.RY(params["weights"][count], wires=q)
                    count = count + 1
                qml.broadcast(
                    qml.CNOT,
                    wires=range(self.n_qubits),
                    pattern=[
                        ((i + 2**layer), i)
                        for i in range(0, self.n_qubits, 2 ** (layer + 1))
                    ],
                )
            qml.RY(params["weights"][count], wires=0)
            return qml.expval(qml.PauliZ(wires=0))

        self.circuit = circuit

        if self.jit:
            circuit = jax.jit(circuit)

        def circuit_plus_bias(params, x):
            return circuit(params, x) + params["bias"]

        self.forward = jax.vmap(circuit_plus_bias, in_axes=(None, 0))
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

        if n_features == 1:
            self.n_qubits = 1
        else:
            n_qubits_ae = int(
                np.ceil(np.log2(n_features))
            )  # the num qubits needed to amplitude encode
            n_qubits = 2 ** int(
                np.ceil(np.log2(n_qubits_ae))
            )  # the model needs 2**m qubits, for some m
            self.n_qubits = n_qubits

        self.initialize_params()
        self.construct_model()

    def initialize_params(self):
        # initialise the trainable parameters
        weights = (
            2
            * jnp.pi
            * jax.random.uniform(
                shape=(2 * self.n_qubits - 1,), key=self.generate_key()
            )
        )

        bias = 0.1 * jax.random.normal(shape=(1,), key=self.generate_key())
        self.params_ = {"weights": weights, "bias": bias}

    def fit(self, X, y):
        """Fit the model to data X and labels y.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)
            y (np.ndarray): Labels of shape (n_samples,)
        """
        self.initialize(X.shape[1], classes=np.unique(y))

        X = self.transform(X)

        def loss_fn(params, X, y):
            # square loss
            predictions = self.forward(params, X)
            return jnp.mean((predictions - y) ** 2)

        if self.jit:
            loss_fn = jax.jit(loss_fn)
        optimizer = optax.adam
        self.params_ = train(
            self,
            loss_fn,
            optimizer,
            X,
            y,
            self.generate_key,
            convergence_interval=self.convergence_interval,
        )

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
        The feature vectors padded to the next power of 2 and then normalised.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)
        """
        n_features = X.shape[1]
        X = X * self.scaling

        n_qubits_ae = int(
            np.ceil(np.log2(n_features))
        )  # the num qubits needed to amplitude encode
        n_qubits = 2 ** int(
            np.ceil(np.log2(n_qubits_ae))
        )  # the model needs 2**m qubits, for some m
        max_n_features = 2**n_qubits
        n_padding = max_n_features - n_features
        padding = np.ones(shape=(len(X), n_padding)) / max_n_features

        X_padded = np.c_[X, padding]
        X_normalised = np.divide(
            X_padded, np.expand_dims(np.linalg.norm(X_padded, axis=1), axis=1)
        )
        return X_normalised
