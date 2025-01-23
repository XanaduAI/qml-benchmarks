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
import itertools


class BaseGenerator(BaseEstimator):
    """
    A base class for generative models.

    We use Scikit-learn's `BaseEstimator` so that we can take advantage of
    Scikit-learn's hyperparameter search algorithms like `GridSearchCV` or
    `RandomizedSearchCV` for hyperparameter tuning.

    Args:
        dim (int): dimension of the data (i.e. the number of features)
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim

    @abstractmethod
    def initialize(self, x: any = None):
        """
        Initialize the model and create the model parameters.

        Args:
            x: batch of data to use to initialize the model
        """
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
    r"""
    A base class for energy-based generative models with common functionalities.

    The class implements MCMC sampling via the energy function. This is used to sample from the model and to train
    the model via k-contrastive divergence (see eqn (3) of arXiv:2101.03288).

    We use Scikit-learn's `BaseEstimator` so that we can take advantage of
    Scikit-learn's hyperparameter search algorithms like `GridSearchCV` or
    `RandomizedSearchCV` for hyperparameter tuning.

    The parameters of the model are stored in the `params_` attribute. The
    model hyperparameters are explicitly passed to the constructor. See the
    `BaseEstimator` documentation for more details.

    References:
    Yang Song, Diederik P. Kingma
    "How to Train Your Energy-Based Models"
    arXiv:2101.03288

    Args:
        dim (int): dimension of the data (i.e. number of features)
        cdiv_steps (int): number of mcmc steps to perform to estimate the constrastive divergence loss (default 1)
        convergence_interval (int): The number of loss values to consider to decide convergence.
        max_steps (int): Maximum number of training steps. A warning will be raised if training did not converge.
        learning_rate (float): Initial learning rate for training.
        batch_size (int): Size of batches used for computing parameter updates.
        jit (bool): Whether to use just in time compilation.
        random_state (int): Seed used for pseudorandom number generation.
    """

    def __init__(
        self,
        dim: int = None,
        learning_rate=0.001,
        batch_size=32,
        max_steps=10000,
        cdiv_steps=1,
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
        self.partition_function = None

        # Train dependent attributes that the function train in self.fit() sets.
        self.history_: list[float] = None
        self.training_time_: float = None

        # jax transformations of class functions
        self.mcmc_step = jax.jit(self.mcmc_step) if self.jit else self.mcmc_step
        self.batched_mcmc_sample = jax.vmap(
            self.mcmc_sample, in_axes=(None, 0, None, 0)
        )

    def generate_key(self):
        return jax.random.PRNGKey(self.rng.integers(1000000))

    @abstractmethod
    def energy(self, params: dict, x: any) -> float:
        """
        The energy function for the model for a batch of configurations x.
        This function should be implemented by the subclass.

        Args:
            params: model parameters that determine the energy function.
            x: batch of configurations of shape (n_batch, dim) for which to calculate the energy
        Returns:
            energy (Array): Array of energies of shape (n_batch,)
        """
        pass

    def mcmc_step(self, args, i):
        """
        Perform one metropolis hastings steps.
        The format is such that it can be used with jax.lax.scan for fast compilation.
        """
        params, key, x = args
        key1, key2 = jax.random.split(key, 2)

        # flip a random bit
        flip_idx = jax.random.choice(key1, jnp.arange(self.dim))
        flip_config = jnp.zeros(self.dim, dtype="int32")
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

    def compute_partition_function(self):
        """
        computes the partition function. Note this scales exponentially with the number of features and
        is therefore only suitable for small models
        """

        print('computing partition fn...')

        def increment_partition_fn(i, val):
            x = all_bitstrings[i]
            return val + jnp.exp(-self.energy(self.params_, x)[0])

        all_bitstrings = jnp.array(list(itertools.product([0, 1], repeat=self.dim )))

        self.partition_function = jax.lax.fori_loop(0, all_bitstrings.shape[0], increment_partition_fn, 0)

        return self.partition_function

    def probability(self, x):
        """
        Compute the probability of a configuration. Requires computation of partition function and is
        therefore only suitable for small models.
        Args:
            x (np.array): A configuration
        """

        if self.partition_function is None:
            self.compute_partition_function()

        return jnp.exp(-self.energy(self.params_, x)[0]) / self.partition_function

    def probabilities(self):
        """
        Compute all probabilities. Requires computation of partition function and is
        therefore only suitable for small models.
        """
        @jax.jit
        def prob(x):
            return self.probability(x)

        all_bitstrings = jnp.array(list(itertools.product([0, 1], repeat=self.dim)))
        return jnp.array([prob(x) for x in all_bitstrings])

    def sample(self, num_samples, num_steps=1000, max_chunk_size=100):
        """
        Sample configurations starting from a random configuration.
        Each sample is generated by sampling a random configuration and perforning a number of mcmc updates.
        Args:
            num_samples (int): number of samples to draw
            num_steps (int): number of mcmc steps before drawing a sample
            max_chunk_size (int): maximum number of samples to vmap the sampling for at a time (large values
                use significant memory)
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
        num_chunks = num_steps // max_chunk_size + 1
        x_init = jnp.array_split(x_init, num_chunks)
        keys = jnp.array_split(keys, num_chunks)
        configs = []
        for elem in zip(x_init, keys):
            new_configs = self.batched_mcmc_sample(
                self.params_, elem[0], num_steps, elem[1]
            )
            configs.append(new_configs[:, -1])

        configs = jnp.concatenate(configs)
        return configs

    def contrastive_divergence_loss(self, params, X, y, key):
        """
        Implementation of the standard contrastive divergence loss function (see eqn 3 of arXiv:2101.03288).
        Args:
            X (array): batch of training examples
            y (array): not used; should be set to None when training
            key: JAX PRNG key used for MCMC sampling
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

    @abstractmethod
    def score(self, X, y=None) -> any:
        """
        Score function to be used with hyperparameter optimization (larger score => better)

        Args:
            X: Dataset to calculate score for
            y: labels (set to None for generative models to interface with sklearn functionality)
        """
        pass
