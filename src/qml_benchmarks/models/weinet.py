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

from math import factorial, ceil
import pennylane as qml
import numpy as np
import jax
import jax.numpy as jnp
import optax
from sklearn.base import BaseEstimator, ClassifierMixin
from qml_benchmarks.model_utils import train
from qml_benchmarks.model_utils import chunk_vmapped_fn

jax.config.update("jax_enable_x64", True)


class WeiNet(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        filter_name="edge_detect",
        learning_rate=0.1,
        max_steps=10000,
        convergence_interval=200,
        random_state=42,
        max_vmap=None,
        jit=True,
        scaling=1.0,
        dev_type="default.qubit.jax",
        qnode_kwargs={"interface": "jax-jit"},
        batch_size=32,
    ):
        """
        Quantum convolutional neural network from https://arxiv.org/abs/2104.06918v3  (see fig 2 of plots)

        The model has two registers: the ancilliary register and the work register. The ancilliary register is used to
        parameterise a 4 qubit state which in turn controls a number of unitaries that act on the work register, where
        the data is encoded via amplitude encoding.

        The qubits with index -1 and height-1 are then traced out, which is equivalent to a type of pooling. All single
        and double correlators <Z> and <ZZ> are measured, and a linear model on these values is used for classification.

        The plots does not specify the loss: we use the binary cross entropy.

        The input data X should have shape (dataset_size,height*width) and will be reshaped to
        (dataset_size,1, height, width) in the model. We assume height=width.

        Note that in figure 2 of the plots, the Hadamards on the ancilla register have no effect since we trace this
        register out. The effect of this register is then to simply perform  a classical mixture of the unitaries
        Q_i on the work register. For simplicity (and to save qubit numbers), we parameterise this distribution
        via params_['s'] rather than model the ancilla qubits themselves.

        Args:
            filter_name (str): The classical filter that defines the unitaries Q_i. either 'edge_detect', 'smooth', or
                'sharpen'.
            learning_rate (float): Initial learning rate for gradient descent.
            convergence_interval (int): The number of loss values to consider to decide convergence.
            max_steps (int): Maximum number of training steps. A warning will be raised if training did not converge.
            batch_size (int): Size of batches used for computing parameter updates.
            max_vmap (int or None): The maximum size of a chunk to vectorise over. Lower values use less memory.
                must divide batch_size.
            jit (bool): Whether to use just in time compilation.
            dev_type (str): string specifying the pennylane device type; e.g. 'default.qubit'.
            qnode_kwargs (str): the key word arguments passed to the circuit qnode.
            random_state (int): Seed used for pseudorandom number generation.
            scaling (float): Factor by which to scale the input data.
        """
        # attributes that do not depend on data
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.filter_name = filter_name
        self.convergence_interval = convergence_interval
        self.dev_type = dev_type
        self.qnode_kwargs = qnode_kwargs
        self.scaling = scaling
        self.batch_size = batch_size
        self.jit = jit
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.unitaries = []

        if filter_name == "edge_detect":
            self.filter = jnp.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        elif filter_name == "smooth":
            self.filter = jnp.array([[1, 1, 1], [1, 5, 1], [1, 1, 1]]) / 13
        elif filter_name == "sharpen":
            self.filter = jnp.array([[-2, -2, -2], [-2, 32, -2], [-2, -2, -2]]) / 16

        if max_vmap is None:
            self.max_vmap = self.batch_size
        else:
            self.max_vmap = max_vmap

        # data-dependant attributes
        # which will be initialised by calling "fit"
        self.n_qubits_ = None
        self.params_ = None  # Dictionary containing the trainable parameters
        self.height_ = None  # height of image data
        self.width_ = None  # width of image data
        self.circuit = None

    def generate_key(self):
        return jax.random.PRNGKey(self.rng.integers(1000000))

    def construct_unitaries(self):
        """
        Construct the unitaries V' defined in the plots
        """
        self.unitaries = [[None for __ in range(3)] for __ in range(3)]
        for mu in range(3):
            for nu, k in enumerate([-1, 0, 1]):
                V = np.zeros([self.height_, self.width_])
                for i in range(self.height_):
                    V[i, (i + k) % self.height_] = self.filter[nu, mu]
                self.unitaries[nu][mu] = V / self.filter[nu, mu]

    def construct_models(self):
        """
        constructs the 9 circuits used for the convolutional layer (Q_k in the plots).
        """

        # get the operators that are used for prediction. We don't include the self.height_ qubit or the last
        # qubit as per the plots, since there are lost in the pooling layer
        operators = []
        for i in range(self.n_qubits_ - 1):
            if i != int(self.n_qubits_ / 2) - 1:
                operators.append(qml.PauliZ(wires=i))
                for j in range(i + 1, self.n_qubits_ - 1):
                    if j != int(self.n_qubits_ / 2) - 1:
                        operators.append(qml.PauliZ(wires=i) @ qml.PauliZ(wires=j))
        operators.append(qml.Identity(wires=0))

        wires = range(self.n_qubits_)
        dev = qml.device(self.dev_type, wires=wires)
        circuits = []
        for nu in range(3):
            for mu in range(3):

                @qml.qnode(dev, **self.qnode_kwargs)
                def circuit(x):
                    qml.AmplitudeEmbedding(
                        jnp.reshape(x, -1), wires=wires, normalize=True, pad_with=0.0
                    )
                    qml.QubitUnitary(
                        jnp.kron(
                            self.unitaries[nu][nu], jnp.array(self.unitaries[nu][mu])
                        ),
                        wires=wires,
                    )
                    return [qml.expval(op) for op in operators]

                self.circuit = (
                    circuit  # we use the last one of the circuits here as an example
                )

                if self.jit:
                    circuit = jax.jit(circuit)
                circuits.append(circuit)

        self.circuits = circuits

    def forward_fn(self, params, x):
        """
        We have taken some shortcuts here compared to the plots description but the result is the same.
        Since we trace out the ancilla register, the final hadamards in the circuit diagram have no effect, and the process
        is equivalent to classically sampling one of the unitaries Q_i, parameterised by params['s'].
        """
        probs = jax.nn.softmax(params["s"])
        expvals = jnp.array(
            [probs[i] * jnp.array(self.circuits[i](x)).T for i in range(9)]
        )
        expvals = jnp.sum(expvals, axis=0)
        out = jnp.sum(params["weights"] * expvals)
        # out = jax.nn.sigmoid(out)  # convert to a probability
        # out = jnp.vstack((out, 1 - out)).T  # convert to 'two neurons'
        # out = jnp.reshape(out, (2))
        return out

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

        im_height = int(jnp.sqrt(n_features))
        self.n_qubits_ = 2 * ceil(jnp.log2(im_height))
        self.height_ = 2 ** (self.n_qubits_ // 2)
        self.width_ = 2 ** (self.n_qubits_ // 2)
        self.initialize_params()
        self.construct_unitaries()
        self.construct_models()

    def initialize_params(self):
        """
        initialise the trainable parameters
        """
        # no of expvals that are combined with weights
        n_expvals = int(
            self.n_qubits_
            - 1
            + factorial(self.n_qubits_ - 2) / 2 / factorial(self.n_qubits_ - 4)
        )

        self.params_ = {
            "s": jax.random.normal(self.generate_key(), shape=(9,)),
            "weights": jax.random.normal(self.generate_key(), shape=(n_expvals,))
            / n_expvals,
        }

    def fit(self, X, y):
        """Fit the model to data X and labels y.

        Args:
            X (np.ndarray): Image data of shape (n_samples, height**2)
            y (np.ndarray): Labels of shape (n_samples,)
        """

        self.initialize(X.shape[1], classes=np.unique(y))
        y = jnp.array(y, dtype=int)
        X = self.transform(X)

        # initialise the model
        self.construct_unitaries()
        self.construct_models()
        self.forward = jax.vmap(self.forward_fn, in_axes=(None, 0))
        self.chunked_forward = chunk_vmapped_fn(self.forward, 1, self.max_vmap)

        def loss_fn(params, X, y):
            # we use the usual cross entropy
            y = jax.nn.relu(y)  # convert to 0,1 labels
            vals = self.forward(params, X)
            loss = jnp.mean(optax.sigmoid_binary_cross_entropy(vals, y))
            return loss

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
        p1 = jax.nn.sigmoid(self.chunked_forward(self.params_, X))
        predictions_2d = jnp.c_[1 - p1, p1]
        return predictions_2d

    def transform(self, X, preprocess=True):
        """
        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)
        """

        # put in NCHW format. We assume square images
        im_height = int(jnp.sqrt(X.shape[1]))
        X = jnp.reshape(X, (X.shape[0], 1, im_height, im_height))

        X = self.scaling * X

        padded_X = np.zeros([X.shape[0], X.shape[1], self.height_, self.width_])
        padded_X[: X.shape[0], : X.shape[1], : X.shape[2], : X.shape[3]] = X
        return jnp.array(padded_X)
