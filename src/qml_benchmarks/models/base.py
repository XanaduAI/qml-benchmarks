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

"""Base classes for models."""

import copy
from abc import abstractmethod

import jax
import jax.numpy as jnp
import numpy as np
import optax
from qml_benchmarks.model_utils import train
from sklearn.base import BaseEstimator


class BaseGenerator(BaseEstimator):
    """
    A base class for generative models.

    We use Scikit-learn's `BaseEstimator` so that we can take advantage of
    Scikit-learn's hyperparameter search algorithms like `GridSearchCV` or
    `RandomizedSearchCV` for hyperparameter tuning.

    Args:
        dim:
            The dimensionality of the samples, e.g., for spins it could be an
            integer or a tuple specifying a grid.
    """

    def __init__(self, dim: int or tuple[int]) -> None:
        self.dim = dim

    @abstractmethod
    def initialize(self, x: any = None):
        """
        Initialize the model and create the model parameters.

        Args:
            x: An example data or dimensionality of the model parameters.
        """
        # self.dim = x.shape[1:]
        pass

    @abstractmethod
    def sample(self, num_samples: int) -> any:
        """
        Sample from the model.

        Args:
            num_samples: The number of samples to generate.
        """
        pass


class EnergyBasedModel(BaseGenerator):
    """
    A base class for energy-based generative models with common functionalities.

    We use Scikit-learn's `BaseEstimator` so that we can take advantage of
    Scikit-learn's hyperparameter search algorithms like `GridSearchCV` or
    `RandomizedSearchCV` for hyperparameter tuning.

    The parameters of the model are stored in the `params_` attribute. The
    model hyperparameters are explicitly passed to the constructor. See the
    `BaseEstimator` documentation for more details.

    References:
    Teh, Yee Whye, Max Welling, Simon Osindero, and Geoffrey E. Hinton.
    "Energy-Based Models for Sparse Overcomplete Representations."
    Journal of Machine Learning Research, vol. 4, 2003, pp. 1235-1260.
    """

    def __init__(
        self,
        dim: int = None,
        learning_rate=0.001,
        batch_size=32,
        max_steps=10000,
        cdiv_steps=100,
        convergence_interval=None,
        random_state=42,
        jit=True,
    ) -> None:
        self.learning_rate = learning_rate
        self.jit = jit
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.cdiv_steps = cdiv_steps
        self.convergence_interval = convergence_interval
        self.vmap = True
        self.max_vmap = None
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        # data depended attributes
        self.params_: dict[str : jnp.array] = None
        self.dim = dim  # initialized to None

        # Train depended attributes that the function train in self.fit() sets.
        # It is not the best practice to set attributes hidden in that function
        # Since it is not clear unless someone knows what the train function does.
        # Therefore we add it here for clarity.
        self.history_: list[float] = None
        self.training_time_: float = None

        self.mcmc_step = jax.jit(self.mcmc_step)
        self.batched_mcmc_sample = jax.vmap(
            self.mcmc_sample, in_axes=(None, 0, None, 0)
        )

    def generate_key(self):
        return jax.random.PRNGKey(self.rng.integers(1000000))

    @abstractmethod
    def energy(self, params: dict, x: any) -> float:
        """
        The energy function for the model for a given configuration x.

        This function should be implemented by the subclass and is also
        responsible for initializing the parameters of the model, if necessary.

        Args:
            x: The configuration to calculate the energy for.
        Returns:
            energy (float): The energy.
        """
        pass

    # TODO: this can be made even more efficient with Numpyro MCMC
    # see qgml.mcmc for a simple example
    def mcmc_step(self, args, i):
        """
        Perform one metropolis hastings steps.
        The format is such that it can be used with jax.lax.scan for fast compilation.
        """
        params, key, x = args
        key1, key2 = jax.random.split(key, 2)
        flip_idx = jax.random.choice(key1, jnp.arange(self.dim))
        flip_config = jnp.zeros(self.dim, dtype='int32')
        flip_config = flip_config.at[flip_idx].set(1)
        x_flip = jnp.array((x + flip_config) % 2)

        en = self.energy(params, jnp.expand_dims(x, 0))[0]

        en_flip = self.energy(params, jnp.expand_dims(x_flip, 0))[0]
        accept_ratio = jnp.exp(-en_flip) / jnp.exp(-en)
        accept = jnp.array(jax.random.bernoulli(key2, accept_ratio), dtype=int)[0]
        x_new = accept * x_flip + (1 - accept) * x
        return [params, key2, x_new], x_new

    def mcmc_sample(self, params, x_init, num_mcmc_steps, key):
        """
        Sample a chain of configurations from a starting configuration x_init
        """
        carry = [params, key, x_init]
        carry, configs = jax.lax.scan(self.mcmc_step, carry, jnp.arange(num_mcmc_steps))
        return configs

    def langevin_sample(self, params, x_init, n_samples, key):
        pass

    def sample(self, num_samples, num_steps=1000, max_chunk_size=100):
        """
        sample configurations starting from a random configuration.
        """
        if self.params_ is None:
            raise ValueError(
                "Model not initialized. Call model.initialize first with"
                "example data sample."
            )
        keys = jax.random.split(self.generate_key(), num_samples)

        x_init = jnp.array(
            jax.random.bernoulli(
                self.generate_key(), p=0.5, shape=(num_samples, self.dim)
            ),
            dtype=int,
        )

        # chunk the sampling, otherwise the vmap can blow the memory
        num_chunks = num_steps//max_chunk_size + 1
        x_init = jnp.array_split(x_init, num_chunks)
        keys = jnp.array_split(keys, num_chunks)
        configs = []
        for elem in zip(x_init, keys):
            new_configs = self.batched_mcmc_sample(self.params_, elem[0], num_steps, elem[1])
            configs.append(new_configs[:,-1])

        configs = jnp.concatenate(configs)
        return configs

    def contrastive_divergence_loss(self, params, X, y, key):
        """
        Contrastive divergence loss function.
        Args:
            X (array): batch of training examples
            y (array): not used; should be set to None when training
            key: jax PRNG key
        """
        keys = jax.random.split(key, X.shape[0])

        # we do not take the gradient wrt the sampling, so decouple the param dict here
        params_copy = copy.deepcopy(params)
        for key in params_copy.keys():
            params_copy[key] = jax.lax.stop_gradient(params_copy[key])

        configs = self.batched_mcmc_sample(params_copy, X, self.cdiv_steps, keys)
        x0 = X
        x1 = configs[:, -1]

        # taking the gradient of this loss is equivalent to the CD-k update
        loss = self.energy(params, x0) - self.energy(params, x1)

        return jnp.mean(loss)

    def fit(self, X: jnp.array, y: any = None) -> None:
        """
        Fit the parameters and update self.params_.
        """
        self.initialize(X)
        c_div_loss = (
            jax.jit(self.contrastive_divergence_loss)
            if self.jit
            else self.contrastive_divergence_loss
        )

        self.params_ = train(
            self,
            c_div_loss,
            optax.adam,
            X,
            None,
            self.generate_key,
            convergence_interval=self.convergence_interval,
        )

    def score(self, X, y=None):
        """Score the model on the given data.

        Higher is better.
        """
        if self.params_ is None:
            self.initialize(X.shape[1])

        c_div_loss = (
            jax.jit(self.contrastive_divergence_loss)
            if self.jit
            else self.contrastive_divergence_loss
        )

        return 1 - c_div_loss(self.params_, X, y, self.generate_key())


class SimpleEnergyModel(EnergyBasedModel):
    """A simple energy-based generative model.

    Example:
    --------
    model = SimpleEnergyModel()

    # Generate random 2D data of 0, 1
    X = np.random.randint(0, 2, size=(100, 2))

    # Initialize and calculate the energy of the model with the given data
    model.initialize(X)
    print(model.energy(model.params_, X))

    # Fit the model to the data
    model.fit(X)

    # Generate 100 samples from the model
    samples = model.sample(100)

    # Score the model on the generated data
    print(model.score(X))
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params_ = None  # Data-dependent attributes

    def initialize(self, x: any = None):
        key = self.generate_key()
        self.dim = x.shape[1]
        initializer = jax.nn.initializers.he_uniform()
        self.params_ = {"weights": initializer(key, (x.shape[1], 1), jnp.float32)}

    def energy(self, params, x):
        # Define the energy function here as the dot product of the parameters.
        return jnp.dot(x, params["weights"])
