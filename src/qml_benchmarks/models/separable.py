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

from sklearn.base import BaseEstimator, ClassifierMixin
import pennylane as qml
import numpy as np
from qml_benchmarks.model_utils import *
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

jax.config.update("jax_enable_x64", True)


class SeparableVariationalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        encoding_layers=1,
        learning_rate=0.001,
        batch_size=32,
        max_vmap=None,
        jit=True,
        max_steps=10000,
        random_state=42,
        scaling=1.0,
        convergence_interval=200,
        dev_type="default.qubit.jax",
        qnode_kwargs={"interface": "jax"},
    ):
        r"""
        Variational model that uses only separable operations (i.e. there is no entanglement in the model). The circuit
        consists of layers of encoding gates and parameterised unitaries followed by measurement of an observable.

        Each encoding layer consists of a trainiable arbitrary qubit rotation on each qubit followed by
        a product angle embedding of the input data, using RY gates. A final layer of trainable qubit rotations is
        applied at the end of the circuit.

        The obserable O is the mean value of Pauli Z observables on each of the output qubits. The value of this
        observable is used to predict the probability for class 1 as :math:`P(+1)=\sigma(6\langle O \rangle)`
        where :math`\sigma` is the logistic funciton. The model is then fit using the cross entropy loss.

        Args:
            encoding_layers (int): number of layers in the data encoding circuit.
            learning_rate (float): learning rate for gradient descent.
            batch_size (int): Size of batches used for computing parameter updates.
            max_vmap (int or None): The maximum size of a chunk to vectorise over. Lower values use less memory.
                must divide batch_size.
            jit (bool): Whether to use just in time compilation.
            convergence_interval (int): The number of loss values to consider to decide convergence.
            max_steps (int): Maximum number of training steps. A warning will be raised if training did not converge.
            dev_type (str): string specifying the pennylane device type; e.g. 'default.qubit'.
            qnode_kwargs (str): the keyword arguments passed to the circuit qnode.
            random_state (int): Seed used for pseudorandom number generation
        """
        # attributes that do not depend on data
        self.encoding_layers = encoding_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.dev_type = dev_type
        self.qnode_kwargs = qnode_kwargs
        self.jit = jit
        self.convergence_interval = convergence_interval
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

        dev = qml.device(self.dev_type, wires=1)

        @qml.qnode(dev, **self.qnode_kwargs)
        def single_qubit_circuit(weights, x):
            """
            To avoid using large state vectors we define the circuit for a single qubit and combine
            below to get the full circuit
            """
            for layer in range(self.encoding_layers):
                qml.Rot(weights[layer, 0], weights[layer, 1], weights[layer, 2], wires=0)
                qml.RY(x, wires=0)
            qml.Rot(
                weights[self.encoding_layers, 0],
                weights[self.encoding_layers, 1],
                weights[self.encoding_layers, 2],
                wires=0,
            )
            return qml.expval(qml.PauliZ(wires=0))

        self.circuit = single_qubit_circuit

        def circuit(params, x):
            return jnp.mean(
                jnp.array(
                    [
                        single_qubit_circuit(params["weights"][i], x[i])
                        for i in range(self.n_qubits_)
                    ]
                )
            )

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

        self.n_qubits_ = n_features
        self.initialize_params()
        self.construct_model()

    def initialize_params(self):
        # initialise the trainable parameters
        weights = (
            2
            * jnp.pi
            * jax.random.uniform(
                shape=(self.n_qubits_, self.encoding_layers + 1, 3),
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
        self.initialize(X.shape[1], np.unique(y))

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
        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)
        """
        if preprocess:
            if self.scaler is None:
                self.scaler = MinMaxScaler(feature_range=(-np.pi / 2, np.pi / 2))
                self.scaler.fit(X)
            X = self.scaler.transform(X)
        return X * self.scaling


class SeparableKernelClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        encoding_layers=1,
        svm=SVC(kernel="precomputed", probability=True),
        C=1.0,
        jit=True,
        random_state=42,
        scaling=1.0,
        dev_type="default.qubit.jax",
        qnode_kwargs={"interface": "jax", "diff_method": None},
    ):
        r"""
        A kernel model that uses a separable embedding. The embedding consists of layers of fixed single qubit X
        rotations by an angle :math:`pi/4` on the Bloch sphere and single qubit Y rotation data encoding gates.

        The kernel is given by

        .. math::
            k(x,x')=\vert\langle 0 \vert U^\dagger(x')U(x)\vert 0 \rangle\vert^2


        Args:
            encoding_layers (int): number of layers in the data encoding circuit.
            svm (sklearn.svm.SVC): scikit-learn SVC class object used to fit the model from the kernel matrix.
            C (float): regularization parameter for the SVC. Lower values imply stronger regularization.
            jit (bool): Whether to use just in time compilation.
            random_state (int): Seed used for pseudorandom number generation.
            dev_type (str): string specifying the pennylane device type; e.g. 'default.qubit'.
            qnode_kwargs (str): the keyword arguments passed to the circuit qnode.
            scaling (float): Factor by which to scale the input data.
        """
        # attributes that do not depend on data
        self.encoding_layers = encoding_layers
        self.dev_type = dev_type
        self.qnode_kwargs = qnode_kwargs
        self.jit = jit
        self.svm = svm
        self.C = C
        self.scaling = scaling
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        # data-dependant attributes
        # which will be initialised by calling "fit"
        self.params_ = None  # Dictionary containing the trainable parameters
        self.n_qubits_ = None
        self.scaler = None  # data scaler will be fitted on training data
        self.circuit = None

    def generate_key(self):
        return jax.random.PRNGKey(self.rng.integers(1000000))

    def construct_circuit(self):
        projector = np.zeros([2, 2])
        projector[0, 0] = 1.0

        dev = qml.device(self.dev_type, wires=1)

        @qml.qnode(dev, **self.qnode_kwargs)
        def qubit_circuit(x):
            """
            Compute the overlap probability for a single qubit to avoid large state vectors.
            This is used to compute the full overlap below.
            """
            for layer in range(self.encoding_layers):
                qml.RX(-np.pi / 4, wires=0)
                qml.RY(-x[0], wires=0)
            for layer in range(self.encoding_layers):
                qml.RY(x[1], wires=0)
                qml.RX(np.pi / 4, wires=0)
            return qml.expval(qml.Hermitian(projector, wires=0))

        self.circuit = qubit_circuit

        def circuit(x):
            probs = [
                qubit_circuit(jnp.array([x[i], x[i + self.n_qubits_]]))
                for i in range(self.n_qubits_)
            ]
            return jnp.prod(jnp.array(probs))

        if self.jit:
            circuit = jax.jit(circuit)
        self.forward = circuit

        return self.forward

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
        Z = np.array([np.concatenate((X1[i], X2[j])) for i in range(dim1) for j in range(dim2)])
        self.construct_circuit()
        kernel_values = [self.forward(z) for z in Z]
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
        self.params_ = {}
        self.construct_circuit()

    def fit(self, X, y):
        """Fit the model to data X and labels y. Uses sklearn's SVM classifier

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)
            y (np.ndarray): Labels of shape (n_samples,)
        """

        self.svm.random_state = int(
            jax.random.randint(self.generate_key(), shape=(1,), minval=0, maxval=1000000)
        )

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
