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

"""Energy-based models for generative modeling."""

import numpy as np
import jax
import jax.numpy as jnp
from qml_benchmarks.model_utils import train
import optax
import copy
import flax.linen as nn

class MLP(nn.Module):
    """
    Simple multilayer perceptron neural network used for the energy model.
    """
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(16)(x)
        x = nn.tanh(x)
        x = nn.Dense(1)(x)
        x = nn.tanh(x)
        return x

class EnergyBasedModel():
    """
    Energy-based model for generative learning.
    The model takes as input energy model written as a flax neural network and uses k contrastive divergence
    to fit the parameters.

    Args:
        learning_rate (float): The learning rate for the CD-k updates
        cdiv_steps (int): The number of sampling steps used in contrastive divergence
        jit (bool): Whether to use just-in-time complilation
        batch_size (int): Size of batches used for computing parameter updates
        max_steps (int): Maximum number of training steps.
        convergence_interval (int or None): The number of loss values to consider to decide convergence.
            If None, training runs until the maximum number of steps. Recommoneded to set to None since
            CD-k does not follow the gradient of a fucntion.
        random_state (int): Seed used for pseudorandom number generation.
    """

    def __init__(self, energy_model=MLP, learning_rate=0.001, cdiv_steps=1, jit=True, batch_size=32,
                 max_steps=200, convergence_interval=None, random_state=42):
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.jit = jit
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.convergence_interval = convergence_interval
        self.cdiv_steps = cdiv_steps
        self.vmap = True
        self.max_vmap = None

        # data depended attributes
        self.params_ = None
        self.n_visible_ = None

        self.mcmc_step = jax.jit(self.mcmc_step)
        self.energy_model = energy_model()

    def generate_key(self):
        return jax.random.PRNGKey(self.rng.integers(1000000))

    def energy(self, params, x):
        """
        The energy function for the model for a given configuration x.

        Args:
            x: The configuration to calculate the energy for.
        Returns:
            energy (float): The energy.
        """
        return self.energy_model.apply(params, x)

    def initialize(self, n_features):
        self.n_visible_ = n_features
        x = jax.random.normal(self.generate_key(), shape=(1, n_features))
        self.params_ = self.energy_model.init(self.generate_key(), x)

    def mcmc_step(self, args, i):
        """
        Perform one metropolis hastings steps.
        The format is such that it can be used with jax.lax.scan for fast compilation.
        """
        params = args[0]
        key = args[1]
        x = args[2]
        key1, key2 = jax.random.split(key, 2)
        flip_idx = jax.random.choice(key1, jnp.arange(self.n_visible_))
        flip_config = jnp.zeros(self.n_visible_, dtype=int)
        flip_config = flip_config.at[flip_idx].set(1)
        x_flip = jnp.array((x + flip_config) % 2)
        en = self.energy(params, x)
        en_flip = self.energy(params, x_flip)
        accept_ratio = jnp.exp(-en_flip) / jnp.exp(-en)
        accept = jnp.array(jax.random.bernoulli(key2, accept_ratio), dtype=int)[0]
        x_new = accept * x_flip + (1 - accept) * x
        return [params, key2, x_new], x

    def mcmc_sample(self, params, x_init, n_samples, key):
        """
        Sample a chain of configurations from a starting configuration x_init
        """
        carry = [params, key, x_init]
        carry, configs = jax.lax.scan(self.mcmc_step, carry, jnp.arange(n_samples))
        return configs

    def langevin_sample(self, params, x_init, n_samples, key):
        pass

    def sample(self, n_samples):
        """
        sample configurations starting from a random configuration.
        """
        key = self.generate_key()
        x_init = jnp.array(jax.random.bernoulli(key, p=0.5, shape=(self.n_visible_,)), dtype=int)
        samples = self.mcmc_sample(self.params_, x_init, n_samples, self.generate_key())
        return jnp.array(samples)

    def fit(self, X):
        """
        Fit the parameters using contrastive divergence
        """
        self.initialize(X.shape[-1])
        X = jnp.array(X, dtype=int)

        # batch the relevant functions
        batched_mcmc_sample = jax.vmap(self.mcmc_sample, in_axes=(None, 0, None, 0))
        batched_energy = jax.vmap(self.energy, in_axes=(None, 0))

        def c_div_loss(params, X, y, key):
            """
            contrastive divergence loss
            Args:
                params (dict): parameter dictionary
                X (array): batch of training examples
                y (array): not used; should be set to None when training
                key: jax PRNG key
            """
            keys = jax.random.split(key, X.shape[0])

            # we do not take the gradient wrt the sampling, so decouple the param dict here
            params_copy = copy.deepcopy(params)
            for key in params_copy.keys():
                params_copy[key] = jax.lax.stop_gradient(params_copy[key])

            configs = batched_mcmc_sample(params_copy, X, self.cdiv_steps + 1, keys)
            x0 = configs[:, 0]
            x1 = configs[:, -1]

            # taking the gradient of this loss is equivalent to the CD-k update
            loss = batched_energy(params, x0) - batched_energy(params, x1)

            return jnp.mean(loss)

        c_div_loss = jax.jit(c_div_loss) if self.jit else c_div_loss

        self.params_ = train(self, c_div_loss, optax.adam, X, None, self.generate_key,
                             convergence_interval=self.convergence_interval)


