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

jax.config.update("jax_enable_x64", True)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.extmath import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from qml_benchmarks.model_utils import chunk_vmapped_fn


class QuantumKitchenSinks(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        linear_model=LogisticRegression(penalty=None, solver="lbfgs", tol=10e-4),
        n_episodes=100,
        n_qfeatures="full",
        var=1.0,
        jit=True,
        max_vmap=None,
        dev_type="default.qubit",
        qnode_kwargs={"interface": "jax", "diff_method": None},
        scaling=1.0,
        random_state=42,
    ):
        r"""
        Quantum kitchen sinks model classical data classification.

        Based on: https://arxiv.org/pdf/1806.08321.pdf

        The quantum computer is used to generate random feature vectors which are
        fed to a linear classifier that we implement using scikit-learn's LogisticRegression (note logistic
        regression is a standard method to perform linear classification despite the name)

        The feature map procedure works as follows:

        1. Linearly transform an input feature vector :math:`x` of length :math:`n`
        via :math:`x' = \omega x + \beta` using random :math:`\omega, \beta`.
        The output is of shape `(n_episodes, n_qfeatures)`.

        2. Feed each row vector of the matrix :math:`x'` into a quantum circuit that
        returns `n_qfeatures` samples. The samples are concatenated in a feature vector :math:`x`
        of length `n_episodes*n_qfeatures`, which is the input to the linear model.

        .. note::

            It is not stated in the plots how to generalise the circuits to higher qubit numbers;
            this implementation is a simple generalisation of the circuit in Fig 2(c), which consists
            of encoding the features into single qubits via X rotaiton gates, and perfoming a sequence
            of CNOT gates on nearest neighbour and next-nearest neighbour qubits.

        Args:
            linear_model (sklearn Estimator): linear model to use with the transformed features
            n_episodes (int): Number of features fed into the linear model after data transformation.
            n_qfeatures (str, int): Determines the number of features fed into the quantum circuit to transform the
                data. This is the number of qubits used by the model. If 'full', the number of qubits is equal
                to the number of input features, if 'half' it is half the number of input features.
            var (float): detemined the variance of the matrix `\omega` used to lienarly transform the input
                features.
            max_vmap (int or None): The maximum size of a chunk to vectorise over. Lower values use less memory.
                must divide batch_size.
            jit (bool): Whether to use just in time compilation.
            dev_type (str): string specifying the pennylane device type; e.g. 'default.qubit'
            qnode_kwargs (str): the key word arguments passed to the circuit qnode.
            random_state (int): Seed used for pseudorandom number generation.
        """

        # attributes that do not depend on data
        self.linear_model = linear_model
        self.n_episodes = n_episodes
        self.n_qfeatures = n_qfeatures
        self.var = var
        self.jit = jit
        self.dev_type = dev_type
        self.qnode_kwargs = qnode_kwargs
        self.scaling = scaling
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        if max_vmap is None:
            self.max_vmap = 100000
        else:
            self.max_vmap = max_vmap

        # data-dependant attributes
        # which will be initialised by calling "fit"
        self.params_ = None
        self.n_qubits_ = None
        self.scaler = None  # data scaler for inputs
        self.scaler2 = None  # data scaler for quantum generated features
        self.circuit = None

    def generate_key(self):
        return jax.random.PRNGKey(self.rng.integers(1000000))

    def construct_model(self):
        """
        The circuit used to generate the feature vectors
        It is not clear from the plots how to generalise this to higher qubit numbers.
        This is a simple generalisation of the circuit in Fig 2(c).
        """

        pattern = [[i, i + 2] for i in range(0, self.n_qubits_ - 2)]

        dev = qml.device(
            self.dev_type,
            wires=self.n_qubits_,
            shots=1,
            prng_key=self.generate_key(),
        )

        @qml.qnode(dev, **self.qnode_kwargs)
        def circuit(Q):

            for i, q in enumerate(Q):
                qml.RX(q, wires=i)
            qml.broadcast(qml.CNOT, wires=range(self.n_qubits_), pattern="double")
            qml.broadcast(qml.CNOT, wires=range(self.n_qubits_), pattern=pattern)

            return qml.sample(wires=range(self.n_qubits_))

        self.circuit = circuit

        if self.jit:
            circuit = jax.jit(circuit)
        circuit = chunk_vmapped_fn(jax.vmap(circuit), 0, self.max_vmap)

        self.forward = circuit

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

        if self.n_qfeatures == "full":
            self.n_qubits_ = n_features
        elif self.n_qfeatures == "half":
            self.n_qubits_ = int(np.ceil(n_features / 2))
        else:
            self.n_qubits_ = int(self.n_qfeatures)

        self.construct_model()
        self.initialize_params()

    def initialize_params(self):
        # initialise the parameters
        omegas = jax.random.normal(
            key=self.generate_key(),
            shape=(self.n_episodes, self.n_qubits_, self.n_features_),
        ) * np.sqrt(self.var)
        betas = (
            2
            * np.pi
            * jax.random.uniform(
                key=self.generate_key(), shape=(self.n_episodes, self.n_qubits_)
            )
        )
        self.params_ = {"omegas": np.array(omegas), "betas": np.array(betas)}

    def fit(self, X, y):
        """Fit the model to data X and labels y.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)
            y (np.ndarray): Labels 1-, 1 of shape (n_samples,)
        """

        self.linear_model.random_state = self.rng.integers(100000)

        self.initialize(X.shape[1], np.unique(y))

        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler.fit(X)
        features = self.transform(X)

        start = time.time()
        self.linear_model.fit(features, y)
        end = time.time()
        self.params_["weights"] = self.linear_model.coef_
        self.training_time_ = end - start
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
        """Predict label probabilities for data X.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)

        Returns:
            y_pred_proba (np.ndarray): Predicted label probabilities of shape
            (n_samples, n_classes)
        """
        X = self.transform(X)
        predictions_2d = self.linear_model.predict_proba(X)
        return softmax(predictions_2d, copy=False)

    def transform(self, X, preprocess=True):
        """
        Apply the feature map: The inputs go through a random linear transformation
        followed by a quantum circuit.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)
        """
        if (
            self.params_["betas"] is None
            or self.params_["omegas"] is None
            or self.circuit is None
        ):
            raise ValueError("Model cannot predict without fitting to data first.")

        if preprocess:
            if self.scaler is None:
                # if the model is unfitted, initialise the scaler here
                self.scaler = MinMaxScaler(feature_range=(-1, 1))
                self.scaler.fit(X)
            X = self.scaler.transform(X)
        X = X * self.scaling

        n_data = X.shape[0]
        input_features = np.zeros([self.n_episodes, n_data, self.n_qubits_])
        for e in range(self.n_episodes):
            stacked_beta = np.stack([self.params_["betas"][e] for __ in range(n_data)])
            input_features[e] = (self.params_["omegas"][e] @ X.T + stacked_beta.T).T
        input_features = np.reshape(input_features, (n_data * self.n_episodes, -1))

        features = self.forward(input_features)
        features = np.reshape(features, (self.n_episodes, n_data, -1))
        features = np.array([features[:, i, :] for i in range(n_data)])
        features = np.reshape(features, (n_data, -1))

        if self.scaler2 is None:
            self.scaler2 = StandardScaler().fit(features)

        return features
