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

from math import ceil
import pennylane as qml
import numpy as np
import jax
from jax import numpy as jnp
import optax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import MinMaxScaler
from qml_benchmarks.model_utils import train
from qml_benchmarks.model_utils import chunk_vmapped_fn

jax.config.update("jax_enable_x64", True)


class DataReuploadingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_layers=4,
        observable_type="single",
        convergence_interval=200,
        max_steps=10000,
        learning_rate=0.05,
        batch_size=32,
        max_vmap=None,
        jit=True,
        scaling=1.0,
        dev_type="default.qubit",
        qnode_kwargs={"interface": "jax-jit"},
        random_state=42,
    ):
        r"""
        Data reuploading classifier from https://arxiv.org/abs/1907.02085. Here we code the 'multi-qubit classifier'.
        The model consists of layers of trainable data encoding unitaries. Each subsequent sequence of 3 elements of
        an input vector x are encoded into a SU(2) rotation via pennylane.Rot. The model therefore uses ceil(d/3)
        qubits.

        The cost function is given by eqn (23) in the plots, and includes trainable parameters for scaling input data
        (omegas), the class weights (alphas), and the circuit parameters (thetas). The cost
        function is a weighted sum of the squared differences between the ideal fidelity to the corresponding class state
        (here, either |0> or |1>) and the output state of the circuit, for a number of qubits specified by
        observable_weight. More specifically

        .. math::

            \ell(\vec{\theta},\vec{\omega},\vec{\alpha},\vec{x}_i) =
            \sum_{j=1}^{n_{\text{max}}}(\alpha_j^0F^0_j-(1-y_i))^2 + (\alpha_j^1F^1_j-y_i)^2

        where :math:`n_max` is given by observable_weight and :math:`F^0_j` is the fidelity of the jth output
        quibt to the state |0>.

        When observable_weight !=1 it is not clear from the plots how to predict labels. Here we look at the average
        fidelities to |0> and |1> of the qubits up to index self.observable_weight and take the label with the largest
        average fidelity. For self.observable_weight=1 this reverts to their strategy in the plots.

        Where possible, we have followed the numerical examples found in the repo which accompanies the plots:

        https://github.com/AdrianPerezSalinas/universal_qlassifier/tree/354d70f940ea737192de4063ff72f859c77e5760

        Args:
            n_layers (int): Number of blocks used in the trainable embedding. The data is uploaded n_layers+1 times.
            observable_type (str): Defines the number of qubits used to evaluate the weighed cost fucntion,
                either 'single', 'half' or 'full'.
                 max_steps (int): Maximum number of training steps. A warning will be raised if training did not
                    converge.
            learning_rate (float): Initial learning rate for training.
            convergence_interval (int): The number of loss values to consider to decide convergence.
            batch_size (int): Size of batches used for computing paraeeter updates.
            max_vmap (int or None): The maximum size of a chunk to vectorise over. Lower values use less memory.
                must divide batch_size.
            jit (bool): Whether to use just in time compilation.
            dev_type (str): string specifying the pennylane device type; e.g. 'default.qubit'.
            qnode_kwargs (str): the keyword arguments passed to the circuit qnode.
            scaling (float): Factor by which to scale the input data.
            random_state (int): Seed used for pseudorandom number generation.
        """

        # attributes that do not depend on data
        self.n_layers = n_layers
        self.observable_type = observable_type
        self.convergence_interval = convergence_interval
        self.max_steps = max_steps
        self.learning_rate = learning_rate
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
        self.params_ = None  # dictionary containing the trainable parameters
        self.n_qubits_ = None
        self.scaler = None  # data preprocessing
        self.circuit = None

    def generate_key(self):
        return jax.random.PRNGKey(self.rng.integers(100000))

    def construct_model(self):
        """Construct the quantum circuit used in the model."""

        dev = qml.device(self.dev_type, wires=self.n_qubits_)

        @qml.qnode(dev, **self.qnode_kwargs)
        def circuit(params, x):
            """A variational quantum circuit with data reuploading.

            Args:
                params (array[float]): dictionary of parameters
                x (array[float]): input vector

            Returns:
                list: Expectation values of Pauli Z on each qubit
            """
            for layer in range(self.n_layers):
                x_idx = 0  # to keep track of the data index
                for i in range(self.n_qubits_):
                    # scaled inputs
                    angles = (
                        x[x_idx : x_idx + 3]
                        * params["omegas"][layer, x_idx : x_idx + 3]
                    )
                    qml.Rot(*angles, wires=i)

                    # variational
                    angles = params["thetas"][i, layer, :]
                    qml.Rot(*angles, wires=i)

                    x_idx += 3
                if layer % 2 == 0:
                    qml.broadcast(qml.CZ, range(self.n_qubits_), pattern="double")
                else:
                    qml.broadcast(qml.CZ, range(self.n_qubits_), pattern="double_odd")

            # final reupload without CZs
            x_idx = 0
            for i in range(self.n_qubits_):
                angles = (
                    x[x_idx : x_idx + 3]
                    * params["omegas"][self.n_layers, x_idx : x_idx + 3]
                    + params["thetas"][i, self.n_layers, :]
                )
                qml.Rot(*angles, wires=i)
                x_idx += 3

            return [qml.expval(qml.PauliZ(wires=[i])) for i in range(self.n_qubits_)]

        self.circuit = circuit

        def circuit_as_array(params, x):
            # pennylane returns a list in the circuit above, so we need this
            return jnp.array(circuit(params, x))

        if self.jit:
            circuit_as_array = jax.jit(circuit_as_array)
        self.forward = jax.vmap(circuit_as_array, in_axes=(None, 0))
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
        self.n_qubits_ = ceil(self.n_features / 3)

        if self.observable_type == "single":
            self.observable_weight = 1
        elif self.observable_type == "half":
            if self.n_qubits_ == 1:
                self.observable_weight = 1
            else:
                self.observable_weight = self.n_qubits_ // 2
        elif self.observable_type == "full":
            self.observable_weight = self.n_qubits_

        self.initialize_params()
        self.construct_model()

    def initialize_params(self):
        # This initialization is the same as found here:
        # https://github.com/AdrianPerezSalinas/universal_qlassifier/blob/master/problem_gen.py
        thetas = jax.random.uniform(
            self.generate_key(), shape=(self.n_qubits_, self.n_layers + 1, 3)
        )
        omegas = jax.random.uniform(
            self.generate_key(), shape=(self.n_layers + 1, self.n_qubits_ * 3)
        )
        alphas = jnp.ones(shape=(2, self.observable_weight))
        self.params_ = {"thetas": thetas, "omegas": omegas, "alphas": alphas}

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

        def loss_fn(params, x, y):
            y_mat = jnp.vstack(
                [y for __ in range(self.observable_weight)]
            ).T  # repeated columns of labels
            y_mat0 = (1 - y_mat) / 2
            y_mat1 = (1 + y_mat) / 2
            alpha_mat0 = jnp.vstack(
                [params["alphas"][0, :] for __ in range(len(x))]
            )  # repeated rows of alpha parameters
            alpha_mat1 = jnp.vstack([params["alphas"][1, :] for __ in range(len(x))])

            expvals = self.forward(params, x)
            probs0 = (1 - expvals[:, : self.observable_weight]) / 2  # fidelity with |0>
            probs1 = (1 + expvals[:, : self.observable_weight]) / 2  # fidelity with |1>
            loss = (
                1
                / 2
                * (
                    jnp.sum(
                        (alpha_mat0 * probs0 - y_mat0) ** 2
                        + (alpha_mat1 * probs1 - y_mat1) ** 2
                    )
                )
            )  # eqn 23 in plots
            return loss / len(y)

        if self.jit:
            loss_fn = jax.jit(loss_fn)
        self.construct_model()
        self.initialize_params()
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
        """Predict labels for batch of data X.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)
        """
        check_is_fitted(self)
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
        predictions = jnp.mean(
            predictions[:, : self.observable_weight], axis=1
        )  # use mean of expvals over relevant qubits
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

        X = X * self.scaling
        X = jnp.pad(X, ((0, 0), (0, (3 - X.shape[1]) % 3)), "constant")
        return X


class DataReuploadingClassifierNoScaling(DataReuploadingClassifier):

    def construct_model(self):
        """Construct the quantum circuit used in the model."""

        dev = qml.device(self.dev_type, wires=self.n_qubits_)

        @qml.qnode(dev, **self.qnode_kwargs)
        def circuit(params, x):
            """A variational quantum circuit with data reuploading.

            Args:
                params (array[float]): dictionary of parameters
                x (array[float]): input vector

            Returns:
                list: Expectation values of Pauli Z on each qubit
            """
            for layer in range(self.n_layers):
                x_idx = 0  # to keep track of the data index
                for i in range(self.n_qubits_):
                    angles = x[x_idx : x_idx + 3] + params["thetas"][i, layer, :]
                    qml.Rot(*angles, wires=i)
                    x_idx += 3
                if layer % 2 == 0:
                    qml.broadcast(qml.CZ, range(self.n_qubits_), pattern="double")
                else:
                    qml.broadcast(qml.CZ, range(self.n_qubits_), pattern="double_odd")

            # final reupload without CZs
            x_idx = 0
            for i in range(self.n_qubits_):
                angles = x[x_idx : x_idx + 3] + params["thetas"][i, self.n_layers, :]
                qml.Rot(*angles, wires=i)
                x_idx += 3

            return [qml.expval(qml.PauliZ(wires=[i])) for i in range(self.n_qubits_)]

        self.circuit = circuit

        def circuit_as_array(params, x):
            # pennylane returns a list in the circuit above, so we need this
            return jnp.array(circuit(params, x))

        if self.jit:
            circuit_as_array = jax.jit(circuit_as_array)
        self.forward = jax.vmap(circuit_as_array, in_axes=(None, 0))
        self.chunked_forward = chunk_vmapped_fn(self.forward, 1, self.max_vmap)

        return self.forward

    def initialize_params(self):
        # This initialization is the same as found here:
        # https://github.com/AdrianPerezSalinas/universal_qlassifier/blob/master/problem_gen.py
        thetas = jax.random.uniform(
            self.generate_key(), shape=(self.n_qubits_, self.n_layers + 1, 3)
        )

        alphas = jnp.ones(shape=(2, self.observable_weight))
        self.params_ = {"thetas": thetas, "alphas": alphas}


class DataReuploadingClassifierNoTrainableEmbedding(DataReuploadingClassifier):

    def construct_model(self):
        """Construct the quantum circuit used in the model."""

        dev = qml.device(self.dev_type, wires=self.n_qubits_)

        @qml.qnode(dev, **self.qnode_kwargs)
        def circuit(params, x):
            """A variational quantum circuit with data reuploading.

            Args:
                params (array[float]): dictionary of parameters
                x (array[float]): input vector

            Returns:
                list: Expectation values of Pauli Z on each qubit
            """
            for layer in range(self.n_layers):
                x_idx = 0  # to keep track of the data index
                for i in range(self.n_qubits_):
                    angles = (
                        x[x_idx : x_idx + 3]
                        * params["omegas"][layer, x_idx : x_idx + 3]
                    )
                    qml.Rot(*angles, wires=i)
                    x_idx += 3

            # final reupload without CZs
            x_idx = 0
            for i in range(self.n_qubits_):
                angles = (
                    x[x_idx : x_idx + 3]
                    * params["omegas"][self.n_layers, x_idx : x_idx + 3]
                )
                qml.Rot(*angles, wires=i)
                x_idx += 3

            for layer in range(self.n_layers):
                x_idx = 0  # to keep track of the data index
                for i in range(self.n_qubits_):
                    angles = params["thetas"][i, layer, :]
                    qml.Rot(*angles, wires=i)
                    x_idx += 3
                if layer % 2 == 0:
                    qml.broadcast(qml.CZ, range(self.n_qubits_), pattern="double")
                else:
                    qml.broadcast(qml.CZ, range(self.n_qubits_), pattern="double_odd")

                # final reupload without CZs
            x_idx = 0
            for i in range(self.n_qubits_):
                angles = params["thetas"][i, self.n_layers, :]
                qml.Rot(*angles, wires=i)
                x_idx += 3

            return [qml.expval(qml.PauliZ(wires=[i])) for i in range(self.n_qubits_)]

        self.circuit = circuit

        def circuit_as_array(params, x):
            # pennylane returns a list in the circuit above, so we need this
            return jnp.array(circuit(params, x))

        if self.jit:
            circuit_as_array = jax.jit(circuit_as_array)
        self.forward = jax.vmap(circuit_as_array, in_axes=(None, 0))
        self.chunked_forward = chunk_vmapped_fn(self.forward, 1, self.max_vmap)

        return self.forward


class DataReuploadingClassifierNoCost(DataReuploadingClassifier):

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

        def loss_fn(params, X, y):
            expvals = jnp.sum(self.forward(params, X), axis=1)
            probs = (1 - expvals * y) / 2  # the probs of incorrect classification
            return jnp.mean(probs)

        if self.jit:
            loss_fn = jax.jit(loss_fn)
        self.construct_model()
        self.initialize_params()
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


class DataReuploadingClassifierSeparable(DataReuploadingClassifier):

    def construct_model(self):
        """Construct the quantum circuit used in the model."""

        dev = qml.device(self.dev_type, wires=self.n_qubits_)

        @qml.qnode(dev, **self.qnode_kwargs)
        def circuit(params, x):
            """A variational quantum circuit with data reuploading.

            Args:
                params (array[float]): dictionary of parameters
                x (array[float]): input vector

            Returns:
                list: Expectation values of Pauli Z on each qubit
            """
            for layer in range(self.n_layers):
                x_idx = 0  # to keep track of the data index
                for i in range(self.n_qubits_):
                    angles = (
                        x[x_idx : x_idx + 3]
                        * params["omegas"][layer, x_idx : x_idx + 3]
                        + params["thetas"][i, layer, :]
                    )
                    qml.Rot(*angles, wires=i)
                    x_idx += 3

            # final reupload without CZs
            x_idx = 0
            for i in range(self.n_qubits_):
                angles = (
                    x[x_idx : x_idx + 3]
                    * params["omegas"][self.n_layers, x_idx : x_idx + 3]
                    + params["thetas"][i, self.n_layers, :]
                )
                qml.Rot(*angles, wires=i)
                x_idx += 3

            return [qml.expval(qml.PauliZ(wires=[i])) for i in range(self.n_qubits_)]

        self.circuit = circuit

        def circuit_as_array(params, x):
            # pennylane returns a list in the circuit above, so we need this
            return jnp.array(circuit(params, x))

        if self.jit:
            circuit_as_array = jax.jit(circuit_as_array)
        self.forward = jax.vmap(circuit_as_array, in_axes=(None, 0))
        self.chunked_forward = chunk_vmapped_fn(self.forward, 1, self.max_vmap)

        return self.forward
