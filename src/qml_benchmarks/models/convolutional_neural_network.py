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
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from qml_benchmarks.model_utils import train

jax.config.update("jax_enable_x64", True)


def construct_cnn(output_channels, kernel_shape):
    class CNN(nn.Module):
        """
        The convolutional neural network used for classification.
        Args:
            x (jnp.array): batch of input data, should be of size (batch_size, height, width, 1)

        Returns: unnormalized activations of the neuron (to be fed to a sigmoid activation)

        """

        @nn.compact
        def __call__(self, x):
            x = nn.Conv(
                features=output_channels[0], kernel_size=(kernel_shape, kernel_shape)
            )(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
            x = nn.Conv(
                features=output_channels[1], kernel_size=(kernel_shape, kernel_shape)
            )(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
            x = x.reshape((x.shape[0], -1))  # flatten
            x = nn.Dense(features=output_channels[1] * 2)(x)
            x = nn.relu(x)
            x = nn.Dense(features=1)(x)
            return x

    return CNN()


class ConvolutionalNeuralNetwork(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        kernel_shape=3,
        output_channels=[32, 64],
        learning_rate=0.001,
        convergence_interval=200,
        max_steps=10000,
        batch_size=32,
        max_vmap=None,
        jit=True,
        random_state=42,
        scaling=1.0,
    ):
        r"""
        This implements a vanilla convolutional neural network (CNN) two-class classifier with JAX and flax
        (https://github.com/google/flax).

        The structure of the neural network is as follows:

        - a 2D convolutional layer with self.output_channels[0] output channels
        - a max pool layer
        - a 2D convolutional layer with self.output_channels[1] output channels
        - a max pool layer
        - a two layer fully connected feedforward neural network with 2*self.output_channels[1] hidden neurons
            and one output neuron


        The probability of class 1 is given by :math:`P(+1\vert \vec{w},\vec{x}) = \sigma(f(\vec{w}),\vec{x})`
        where :math:`\vec{w}` are the weights of the network and :math:`\sigma` is the logistic function and
        :math:`f` gives the value of the neuron in the final later. These probabilities are fed to binary cross entropy
        loss for training.

        The 2d input data should be flattened to have shape (n_samples, height*width), where height and width are the
        dimensions of the 2d data. The model works for square data only, i.e. height=width.

        Args:
            kernel_shape (int): the shape of the kernel used in the CNN. e.g. kernel_shape=3 uses a 3x3 filter
            output_channels (list[int]): Two integers specifying the output sizes of the convolutional layers
                in the CNN. Defaults to [32, 62].
            learning_rate (float): Initial learning rate for training.
            max_steps (int): Maximum number of training steps. A warning will be raised if training did not converge.
            convergence_interval (int): The number of loss values to consider to decide convergence.
            batch_size (int): Size of batches used for computing parameter updates.
            max_vmap (int or None): The maximum size of a chunk to vectorise over. Lower values use less memory.
                must divide batch_size.
            jit (bool): Whether to use just in time compilation.
            random_state (int): Seed used for pseudorandom number generation.
            scaling (float): Factor by which to scale the input data.
        """

        # attributes that do not depend on data
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.scaling = scaling
        self.jit = jit
        self.kernel_shape = kernel_shape
        self.output_channels = output_channels
        self.convergence_interval = convergence_interval
        self.batch_size = batch_size
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

    def generate_key(self):
        return jax.random.PRNGKey(self.rng.integers(1000000))

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

        # initialise the model
        self.cnn = construct_cnn(self.output_channels, self.kernel_shape)
        self.forward = self.cnn

        # create dummy data input to initialise the cnn
        height = int(jnp.sqrt(n_features))
        X0 = jnp.ones(shape=(1, height, height, 1))
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

        # scale input data
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        X = self.transform(X)

        def loss_fn(params, X, y):
            y = jax.nn.relu(y)  # convert to 0,1 labels
            vals = self.forward.apply(params, X)[:, 0]
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
        # get probabilities of y=1
        p1 = jax.nn.sigmoid(self.forward.apply(self.params_, X)[:, 0])
        predictions_2d = jnp.c_[1 - p1, p1]
        return predictions_2d

    def transform(self, X):
        """
        If scaler is initialized, transform the inputs.

        Put into NCHW format. This assumes square images.
        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)
        """
        if self.scaler is None:
            # if the model is unfitted, initialise the scaler here
            self.scaler = StandardScaler()
            self.scaler.fit(X)

        X = self.scaler.transform(X) * self.scaling

        # reshape data to square array
        X = jnp.array(X)
        height = int(jnp.sqrt(X.shape[1]))
        X = jnp.reshape(X, (X.shape[0], height, height, 1))

        return X
