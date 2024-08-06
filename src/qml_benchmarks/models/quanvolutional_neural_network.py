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

import numpy as np
import pennylane as qml
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from qml_benchmarks.model_utils import chunk_vmapped_fn
from qml_benchmarks.model_utils import train

jax.config.update("jax_enable_x64", True)


def construct_cnn(output_channels, kernel_shape):
    class CNN(nn.Module):
        """
        The convolutional neural network used for classification. The strucurre of the network is the same as the model
        ConvolutionalNeuralNetwork.

        Args:
            x (jnp.array): batch of input data, should be of size (batch_size, height, width, 1)

        Returns: unnormalized activations of the neuron (to be fed to a sigmoid activation)

        """

        @nn.compact
        def __call__(self, x):
            x = nn.Conv(
                features=output_channels[0],
                kernel_size=(kernel_shape, kernel_shape),
                padding="SAME",
            )(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")
            x = nn.Conv(
                features=output_channels[1],
                kernel_size=(kernel_shape, kernel_shape),
                padding="SAME",
            )(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")
            x = x.reshape((x.shape[0], -1))  # flatten
            x = nn.Dense(features=output_channels[1] * 2)(x)
            x = nn.relu(x)
            x = nn.Dense(features=1)(x)

            return x

    return CNN()


class QuanvolutionalNeuralNetwork(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        qkernel_shape=2,
        n_qchannels=1,
        rand_depth=10,
        rand_rot=20,
        threshold=0.0,
        kernel_shape=3,
        output_channels=[32, 64],
        max_vmap=None,
        jit=True,
        learning_rate=0.001,
        max_steps=10000,
        convergence_interval=200,
        batch_size=32,
        random_state=42,
        scaling=1.0,
        dev_type="default.qubit.jax",
        qnode_kwargs={"interface": "jax-jit"},
    ):
        r"""
        Quanvolutional Neural network described in https://arxiv.org/pdf/1904.04767v1.pdf.

        The model is an adaptation of the ConvolutionalNeuralNetwork model. The only difference is that the data is
        preprocessed by a quantum convolutional layer before being fed to the classical CNN. This quantum preprocessing
        is a fixed, non-trainiable feature map.

        The feature map consists of the following:

            1. A step function that converts the input to 0,1 valued. If the value is below `threshold` it is 0,
            otherwise it is 1.

            2. A 2d convolution layer is a applied to the binarised data. The filter is given by a random quantum
            circuit acting on `qkernel_shape*qkernel_shape` qubits (a square grid), and there are `n_qchannels` output
            channels. The scalar output of the filter is given by the number of ones appearing in the output bitstring
            of the circuit with the highest probability to be sampled. We implement the random quantum circuit
            via PennyLane's `RandomLayers`.

            3. The transformed data is fed into a classical CNN of the same form as ConvolutionalNeuralNetwork and
            the model is equivalent from that point onwards.

        The input data X should have shape (dataset_size,height*width) and will be reshaped to
        (dataset_size, height, width, 1) in the model. We assume height=width.

        Args:
            qkernel_shape (int): The size of the quantum filter: a circuit with `qkernel_shape*qkernel_shape`
                qubits will be used.
            n_qchannels (int): The number of output channels in the quanvolutional layer.
            rand_depth (int): The depth of the random circuit in pennylane.RandomLayers
            rand_rot (int): The number of random rotations in pennylane.RandomLayers
            threshold (float): The threshold that determines the binarisation of the input data. Since we use a
                StadardScaler this is set to 0.0 as default.
            kernel_shape (int): the size of the kernel used in the CNN. e.g. kernel_shape=3 uses a 3x3 filter
            output_channels (list[int]): a list of integers specifying the output size of the convolutional layers
                in the CNN.
            max_vmap (int or None): The maximum size of a chunk to vectorise over. Lower values use less memory.
                must divide batch_size.
            jit (bool): Whether to use just in time compilation.
            convergence_interval (int): The number of loss values to consider to decide convergence.
            learning_rate (float): Initial learning rate for training.
            max_steps (int): Maximum number of training steps. A warning will be raised if training did not converge.
            batch_size (int): Size of batches used for computing parameter updates.
            dev_type (str): string specifying the pennylane device type; e.g. 'default.qubit'
            qnode_kwargs (str): the key word arguments passed to the circuit qnode.
            random_state (int): Seed used for pseudorandom number generation.
        """
        # attributes that do not depend on data
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.convergence_interval = convergence_interval
        self.n_qchannels = n_qchannels
        self.rand_depth = rand_depth
        self.rand_rot = rand_rot
        self.threshold = threshold
        self.kernel_shape = kernel_shape
        self.qkernel_shape = qkernel_shape
        self.output_channels = output_channels
        self.batch_size = batch_size
        self.jit = jit
        self.dev_type = dev_type
        self.qnode_kwargs = qnode_kwargs
        self.batch_size = batch_size
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
        self.scaler = None  # data scaler will be fitted on training data
        self.circuit = None

    def generate_key(self):
        return jax.random.PRNGKey(self.rng.integers(1000000))

    def construct_random_circuit(self):
        """
        construct a random circuit to be used as a filter in the quanvolutional layer
        """
        wires = range(self.qkernel_shape**2)
        dev = qml.device(self.dev_type, wires=wires)
        weights = (
            jnp.pi
            * 2
            * jnp.array(
                jax.random.uniform(
                    self.generate_key(), shape=(self.rand_depth, self.rand_rot)
                )
            )
        )

        @qml.qnode(dev, **self.qnode_kwargs)
        def circuit(x):
            """
            Apply a random circuit and return the probabilities of the output strings.
            Here we use Pennylane's RandomLayers which deviates slightly from the desciption in the plots,
            but we expect similar behaviour since they are both random circuit generators.
            """
            for i in wires:
                qml.RY(x[i] * jnp.pi, wires=i)
            qml.RandomLayers(weights, wires=wires)
            return qml.probs(wires=wires)

        self.circuit = circuit

        if self.jit:
            circuit = jax.jit(circuit)
        circuit = chunk_vmapped_fn(jax.vmap(circuit, in_axes=(0)), 0, self.max_vmap)
        return circuit

    def construct_quanvolutional_layer(self):
        """
        construct the quantum feature map.
        """
        random_circuits = [
            self.construct_random_circuit() for __ in range(self.n_qchannels)
        ]

        # construct an array that specifies the indices of the 'windows' of the image used for the convolution.
        idx_mat = jnp.array(
            [[(i, j) for j in range(self.width)] for i in range(self.height)]
        )
        idxs = jnp.array(
            [
                idx_mat[j : j + self.qkernel_shape, k : k + self.qkernel_shape]
                for k in range(self.height - self.qkernel_shape + 1)
                for j in range(self.height - self.qkernel_shape + 1)
            ]
        )
        idxs = idxs.reshape(len(idxs), -1, 2)
        zerovec = jnp.zeros(len(idxs[0, :, 0]))  # needed for last axis of NHWC format
        idxs = jnp.array(
            [[idxs[i, :, 0], idxs[i, :, 1], zerovec] for i in range(len(idxs))],
            dtype=int,
        )

        def quanv_layer(x):
            """
            A convolutional layer where the filter is given by a random quantum circuit. The layer has a stride of 1.
            Args:
                x (jnp.array): input data of shape (n_data, height, width, 1)
            """

            # the windows from the image to be fed into the quantum circuits
            x_windows = x[idxs[:, 0, :], idxs[:, 1, :], idxs[:, 2, :]]

            layer_out = []
            for channel in range(self.n_qchannels):
                out = []
                # find most likely outputs
                probs = random_circuits[channel](x_windows)
                # convert to scalars based on the number of ones in the state vec
                max_idxs = jnp.argmax(probs, axis=1)
                state_vecs = [
                    jnp.unravel_index(idx, [2 for __ in range(self.qkernel_shape**2)])
                    for idx in max_idxs
                ]
                out = jnp.sum(jnp.array(state_vecs), axis=1)
                out = jnp.array(out, dtype="float64")
                # put back to correct shape
                out = jnp.reshape(
                    out,
                    (
                        self.height - self.qkernel_shape + 1,
                        self.width - self.qkernel_shape + 1,
                    ),
                )
                layer_out.append(out)

            layer_out = jnp.array(layer_out)
            layer_out = jnp.moveaxis(
                layer_out, 0, -1
            )  # the above is in CHW format, so we switch to WHC
            return layer_out

        return quanv_layer

    def forward(self, params, X):
        X = self.batched_quanv_layer(X)
        out = self.cnn.apply(params, X)
        return out

    def initialize(self, n_features, classes=None):
        """Initialize attributes that depend on the number of features and the class labels.
        Args:
            classes (array-like): class labels that the classifier expects
        """
        if classes is None:
            classes = [-1, 1]

        self.classes_ = classes
        self.n_classes_ = len(self.classes_)
        assert self.n_classes_ == 2
        assert 1 in self.classes_ and -1 in self.classes_

        self.height = int(jnp.sqrt(n_features))
        self.width = self.height

        # initialise the model
        self.quanv_layer = self.construct_quanvolutional_layer()
        self.batched_quanv_layer = chunk_vmapped_fn(
            jax.vmap(self.quanv_layer, in_axes=(0)), 0, self.max_vmap
        )
        self.cnn = construct_cnn(self.output_channels, self.kernel_shape)

        # create dummy data input to initialise the cnn
        X0 = jnp.ones(shape=(1, self.height, self.height, 1))
        X0 = self.batched_quanv_layer(X0)
        self.initialize_params(X0)

    def initialize_params(self, X):
        # initialise the trainable parameters
        self.params_ = self.cnn.init(self.generate_key(), X)

    def fit(self, X, y):
        """Fit the model to data X and labels y.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)
            y (np.ndarray): Labels of shape (n_samples,)
        """

        self.initialize(X.shape[1], classes=np.unique(y))

        y = jnp.array(y, dtype=int)
        self.scaler = MinMaxScaler(feature_range=(-np.pi / 2, np.pi / 2))
        self.scaler.fit(X)
        X = self.transform(X)

        # quantum feature map the entire dataset for training. Assuming we use more than one epoch of training,
        # it is more efficient to do this first
        X = self.batched_quanv_layer(X)

        def loss_fn(params, X, y):
            """
            this takes the quantum feature mapped data as input and returns the sigmoid binary cross entropy
            """
            y = jax.nn.relu(y)  # convert to 0,1 labels
            vals = self.cnn.apply(params, X)[:, 0]
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
        mapped_predictions = jnp.argmax(predictions, axis=1)
        return jnp.take(self.classes_, mapped_predictions)

    def predict_proba(self, X):
        """Predict label probabilities for data X.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)

        Returns:
            y_pred_proba (np.ndarray): Predicted label probabilities of shape
            (n_samples, n_classes)
        """
        X = self.transform(X)
        p1 = jax.nn.sigmoid(self.forward(self.params_, X)[:, 0])
        predictions_2d = jnp.c_[1 - p1, p1]
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
        X = jnp.array(X)

        # put in NHWC format. We assume square images
        self.height = int(jnp.sqrt(X.shape[1]))
        self.width = self.height
        X = jnp.reshape(X, (X.shape[0], self.height, self.width, 1))
        X = jnp.heaviside(X - self.threshold, 0.0)  # binarise input

        return X
