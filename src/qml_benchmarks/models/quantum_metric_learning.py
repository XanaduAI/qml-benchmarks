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

import warnings
import pennylane as qml
import catalyst
from catalyst import qjit
import numpy as np
import jax
from jax import numpy as jnp
import optax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler
from qml_benchmarks.model_utils import chunk_vmapped_fn, train, train_with_catalyst

jax.config.update("jax_enable_x64", True)


def get_batch(batch_size, A, B, keys):
    """Convenience function to get a batch of data."""
    if batch_size > min(len(A), len(B)):
        raise ValueError(
            f"Trying to select {batch_size} of {min(len(A), len(B))} data. "
            f"Is the batch size too low?"
        )
    select_A = jax.random.choice(
        keys[0], jnp.array(range(len(A))), shape=(batch_size,), replace=False
    )
    select_B = jax.random.choice(
        keys[1], jnp.array(range(len(B))), shape=(batch_size,), replace=False
    )
    return A[select_A], B[select_B]


class QuantumMetricLearner(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_layers=3,
        n_examples_predict=32,
        convergence_interval=200,
        max_steps=50000,
        learning_rate=0.01,
        batch_size=32,
        use_jax = True,
        vmap = True,
        max_vmap=None,
        jit=True,
        random_state=42,
        scaling=1.0,
        dev_type=None,
        qnode_kwargs={"interface": "jax-jit"},
    ):
        """
        Following https://arxiv.org/abs/2001.03622.

        This classifier uses a trainable embedding to encode inputs into quantum states. Training and prediction
        relies on comparing these states with each other using the fidelity/state overlap:

            * During training, the embedding is optimised to place data from the same class close together and data
              from different classes far apart from each other.
            * Prediction compares a new embedded input with memorised training samples from each class and predicts
              the class whose samples are closest.

        Since pairwise comparison between data points are expensive, training and classification only
        uses samples from the data.

        The trainable embedding uses PennyLane's `QAOAEmbedding`.

        The classifier uses `batch_size*3` circuits for an evaluation of the loss function, and `n_examples_predict*2`
        circuits for prediction.

        Args:
            n_examples_predict (int): Number of examples from each class of the training set used for prediction.
            n_layers (int): Number of layers used in the trainable embedding.
            convergence_interval (int): The number of loss values to consider to decide convergence.
            max_steps (int): Maximum number of training steps. A warning will be raised if training did not converge.
            batch_size (int): Size of batches used for computing parameter updates.
            learning_rate (float): Initial learning rate for training.
            max_vmap (int or None): The maximum size of a chunk to vectorise over. Lower values use less memory.
                must divide batch_size.
            jit (bool): Whether to use just in time compilation.
            dev_type (str): Wtring specifying the pennylane device type; e.g. 'default.qubit'
            qnode_kwargs (str): The keyword arguments passed to the circuit qnode
            scaling (float): Factor by which to scale the input data.

        """

        # attributes that do not depend on data
        self.n_examples_predict = n_examples_predict
        self.n_layers = n_layers
        self.convergence_interval = convergence_interval
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.jit = jit
        self.use_jax = use_jax
        self.vmap = vmap
        self.qnode_kwargs = qnode_kwargs
        self.scaling = scaling
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        if dev_type is not None:
            self.dev_type = dev_type
        else:
            self.dev_type = "default.qubit.jax" if use_jax else "lightning.qubit"

        self.max_vmap = 4 if max_vmap is None else max_vmap

        # data-dependant attributes
        # which will be initialised by calling "fit"
        self.params_ = None  # Dictionary containing the trainable parameters
        self.n_qubits_ = None
        self.n_features_ = None
        self.scaler = None  # data scaler will be fitted on training data

    def generate_key(self):
        return jax.random.PRNGKey(self.rng.integers(1000000))

    def construct_model(self):
        dev = qml.device(self.dev_type, wires=self.n_qubits_)
        wires = range(self.n_qubits_)

        def wrapped_circuit(params, x1, x2):
            @qml.qnode(dev, **self.qnode_kwargs)
            def circuit(params, x1, x2):
                qml.QAOAEmbedding(features=x1, weights=params["weights"], wires=wires)
                qml.adjoint(qml.QAOAEmbedding(features=x2, weights=params["weights"], wires=wires))
                return qml.probs()
            return circuit(params, x1, x2)[0]

        circuit = wrapped_circuit

        if self.jit:
            if self.use_jax:
                circuit = jax.jit(circuit)
            else:
                qjit(circuit, autograph=True)

        # always vmapping for now
        if self.use_jax:
            batched_overlaps = jax.vmap(circuit, in_axes=(None, 0, 0))
            chunked_overlaps = chunk_vmapped_fn(batched_overlaps, start=1, max_vmap=self.max_vmap)
        else:
            def batched_overlaps(params, X1, X2):
                return jnp.array([circuit(params, elem[0], elem[1]) for elem in jnp.stack((X1, X2), axis=1)])

        def model(params, X1=None, X2=None):
            res = batched_overlaps(params, X1, X2)
            return jnp.mean(res)

        self.forward = model

        if self.use_jax:
            def chunked_model(params, X1=None, X2=None):
                res = chunked_overlaps(params, X1, X2)
                return jnp.mean(res)

            self.chunked_forward = chunked_model
        else:
            self.chunked_forward = self.forward

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

        self.n_features_ = n_features
        self.n_qubits_ = (
            self.n_features_ + 1
        )  # +1 to add constant features as described in the plots
        self.initialize_params()
        self.construct_model()

    def initialize_params(self):
        shape = qml.QAOAEmbedding.shape(n_layers=self.n_layers, n_wires=self.n_qubits_)
        weights = jax.random.normal(key=self.generate_key(), shape=shape) * 0.1
        self.params_ = {"weights": weights}

    def fit(self, X, y):
        """Fit the model to data X and labels y.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)
            y (np.ndarray): Labels of shape (n_samples,)
        """

        self.initialize(X.shape[1], classes=np.unique(y))

        self.scaler = MinMaxScaler(feature_range=(-np.pi / 2, np.pi / 2))
        self.scaler.fit(X)
        X = self.transform(X)

        # split data
        A = jnp.array(X[y == -1])
        B = jnp.array(X[y == 1])

        if self.batch_size > min(len(A), len(B)):
            warnings.warn("batch size too large, setting to " + str(min(len(A), len(B))))
            self.batch_size = min(len(A), len(B))

        def loss_fn(params, A=None, B=None):
            aa = self.forward(params, X1=A, X2=A)
            bb = self.forward(params, X1=B, X2=B)
            ab = self.forward(params, X1=A, X2=B)

            d_hs = -ab + 0.5 * (aa + bb)
            return 1 - d_hs

        if self.jit:
            loss_fn = jax.jit(loss_fn) if self.use_jax else qjit(loss_fn)

        optimizer = optax.adam

        if self.use_jax:
            self.params_ = train(self, loss_fn, optimizer, A, B, self.generate_key,
                                 convergence_interval=self.convergence_interval)

        else:
            self.params_ = train_with_catalyst(self, loss_fn, optimizer, A, B, self.generate_key,
                                               convergence_interval=self.convergence_interval)

        self.params_["examples_-1"] = A
        self.params_["examples_+1"] = B

        return self

    def predict(self, X):
        """Predict labels for batch of data X.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)
        """
        predictions = self.predict_proba(X)
        mapped_predictions = np.argmax(predictions, axis=1)
        return np.take(self.classes_, mapped_predictions)

    def predict_proba(self, X):
        """Predict label probabilities for data X using a batch of training examples.
        The examples are stored in the parameter dictionary.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)

        Returns:
            y_pred_proba (np.ndarray): Predicted label probabilities of shape
            (n_samples, n_classes)
        """
        if "examples_-1" not in self.params_:
            raise ValueError("Model cannot predict without fitting to data first.")

        X = self.transform(X)

        max_examples = min(len(self.params_["examples_-1"]), len(self.params_["examples_+1"]))
        if self.n_examples_predict > max_examples:
            warnings.warn("n_examples_predict too large, setting to " + str(max_examples))
            self.n_examples_predict = max_examples

        A_examples, B_examples = get_batch(
            self.n_examples_predict,
            self.params_["examples_-1"],
            self.params_["examples_+1"],
            [self.generate_key(), self.generate_key()],
        )

        predictions = []
        for x in X:
            # create list [x, x, x, ...] to get overlaps with A_examples = [a1, a2, a3...] and B_examples
            x_tiled = jnp.tile(x, (self.n_examples_predict, 1))

            if self.use_jax:
                pred_a = jnp.mean(self.chunked_forward(self.params_, A_examples, x_tiled))
                pred_b = jnp.mean(self.chunked_forward(self.params_, B_examples, x_tiled))
            else:
                pred_a = jnp.mean(self.forward(self.params_, A_examples, x_tiled))
                pred_b = jnp.mean(self.forward(self.params_, B_examples, x_tiled))

            # normalise to [0,1]
            predictions.append([pred_a / (pred_a + pred_b), pred_b / (pred_a + pred_b)])

        return jnp.array(predictions)

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
