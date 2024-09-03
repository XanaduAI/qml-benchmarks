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

"""Generate a dataset of spin samples from Ising models."""

from collections import namedtuple

import jax
import numpyro
from jax import numpy as jnp
from jax import random
from jax.typing import ArrayLike
from numpy import ndarray
from numpyro.infer import MCMC
from numpyro.infer.mcmc import MCMCKernel
from qml_benchmarks.data.base import SpinGeneratorBase
from tqdm.auto import tqdm


def create_2D_interaction_matrix(grid_size: int) -> ArrayLike:
    """
    Create an interaction matrix for a 2D square lattice with periodic boundary
    conditions.

    This function generates a symmetric interaction matrix for a 2D Ising model
    on a square lattice. Each spin interacts with its four nearest neighbors
    (up, down, left, right), with periodic boundary conditions applied.

    Args:
        grid_size (int):
            The grid size of the square lattice. The total number of spins will
            be grid_size^2.

    Returns:
        ArrayLike:
            A 2D square array of shape (grid_size^2, grid_size^2) representing the
            interaction matrix. Each element J[i, j] is 1 if spins i and j are
            nearest neighbors, and 0 otherwise.

    Notes:
    ------
    The resulting matrix is symmetric and sparse, with exactly four non-zero
    elements in each row/column corresponding to the four nearest neighbors
    of each spin.
    """
    J = jnp.zeros((grid_size * grid_size, grid_size * grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            # Spin index in the grid
            idx = i * grid_size + j

            # Calculate the indices of the neighbors
            right_idx = i * grid_size + (j + 1) % grid_size
            left_idx = i * grid_size + (j - 1) % grid_size
            bottom_idx = ((i + 1) % grid_size) * grid_size + j
            top_idx = ((i - 1) % grid_size) * grid_size + j

            # Set the interactions, ensuring each pair is only added once
            J = J.at[idx, right_idx].set(1)
            J = J.at[idx, left_idx].set(1)
            J = J.at[idx, bottom_idx].set(1)
            J = J.at[idx, top_idx].set(1)
    return J


@jax.jit
def energy(
    sample: ArrayLike, J: ArrayLike, bias: ArrayLike, J_sparse: tuple = None
) -> float:
    """Calculate the energy of a sample of spin configuration.

    Args:
        sample (ArrayLike):
            The spin configuration. A 1D array where each element is either
            +1 or -1.
        J (ArrayLike):
            The interaction matrix. A 2D array where J[i,j] represents the interaction
            strength between spins i and j.
        bias (ArrayLike):
            The bias term. A 1D array where b[i] represents the bias on spin i.
        J_sparse (tuple, optional):
            A sparse representation of the interaction matrix. Defaults to None.

    Returns:
        float: The energy of the sample.
    """
    if J_sparse is not None:
        return -jnp.einsum(
            "i,i,i->", sample[J_sparse[0]], sample[J_sparse[1]], J[J_sparse]
        ) / 2.0 - jnp.dot(sample, bias)

    return -jnp.einsum("i,j,ij->", sample, sample, J) / 2.0 - jnp.dot(sample, bias)


def initialize_spins(rng_key, N: int, num_chains: int) -> ArrayLike:
    """
    Initialize a configuration of spins for an Ising model.

    This function generates initial random spin configurations for one or more
    Markov chains. Each spin is randomly set to either +1 or -1 with equal
    probability.

    Args:
        rng_key (jax.random.PRNGKey):
            A JAX random number generator key.
        N (int):
            The number of spins in each configuration.
        num_chains (int):
            The number of Markov chains (i.e., the number of separate spin
            configurations) to initialize.

    Returns:
        jax.numpy.ndarray:
            An array of initialized spins. If num_chains is 1, the shape is (N,).
            Otherwise, the shape is (num_chains, N). Each element is either +1 or -1.
    """
    if num_chains == 1:
        # this is needed
        spins = random.bernoulli(rng_key, 0.5, (N,))
    else:
        spins = random.bernoulli(
            rng_key,
            0.5,
            (
                num_chains,
                N,
            ),
        )
    return spins * 2 - 1


MHState = namedtuple("MHState", ["spins", "rng_key"])


class MetropolisHastings(MCMCKernel):
    """An implementation of MCMC using Numpyro, see example in
    https://num.pyro.ai/en/stable/mcmc.html
    """

    sample_field = "spins"

    def init(
        self,
        rng_key: random.PRNGKey,
        num_warmup: int,
        init_params: jnp.array,
        *args,
        **kwargs
    ):
        """Initialize the state of the model."""
        return MHState(init_params, rng_key)

    def sample(self, state: jnp.array, model_args, model_kwargs):
        """Sample from the model."""
        spins, rng_key = state
        num_spins = spins.size

        def mh_step(i, val):
            spins, rng_key = val
            rng_key, subkey = random.split(rng_key)
            flip_index = random.randint(subkey, (), 0, num_spins)
            spins_proposal = spins.at[flip_index].set(-spins[flip_index])

            current_energy = energy(
                spins, model_kwargs["J"], model_kwargs["b"], model_kwargs["J_sparse"]
            )
            proposed_energy = energy(
                spins_proposal,
                model_kwargs["J"],
                model_kwargs["b"],
                model_kwargs["J_sparse"],
            )
            delta_energy = proposed_energy - current_energy
            accept_prob = jnp.exp(-delta_energy / model_kwargs["T"])

            rng_key, subkey = random.split(rng_key)
            accept = random.uniform(subkey) < accept_prob
            spins = jnp.where(accept, spins_proposal, spins)
            return spins, rng_key

        spins, rng_key = jax.lax.fori_loop(0, num_spins, mh_step, (spins, rng_key))
        return MHState(spins, rng_key)


class IsingSpins(SpinGeneratorBase):
    def __init__(
        self, N: int, J: jnp.array, b: jnp.array, T: float, sparse=False
    ) -> None:
        super().__init__(N)
        self.kernel = MetropolisHastings()
        self.J = J
        self.T = T
        self.b = b
        self.J_sparse = jnp.nonzero(J) if sparse else None

        if N < 16:
            Z = 0
            for i in tqdm(range(2**self.N), desc="Computing partition function"):
                lattice = (-1) ** jnp.array(jnp.unravel_index(i, [2] * self.N))
                en = energy(lattice, self.J, self.b, self.J_sparse)
                Z += jnp.exp(-en / T)
            self.Z = Z

    def sample(
        self, n_samples: int, num_chains=1, thinning=1, num_warmup=1000, key=42
    ) -> jnp.array:
        rng_key = random.PRNGKey(key)
        init_spins = initialize_spins(rng_key, self.N, num_chains)
        mcmc = MCMC(
            self.kernel,
            num_warmup=num_warmup,
            thinning=thinning,
            num_samples=n_samples,
            num_chains=num_chains,
        )
        mcmc.run(
            rng_key,
            init_params=init_spins,
            J=self.J,
            b=self.b,
            T=self.T,
            J_sparse=self.J_sparse,
        )
        samples = mcmc.get_samples()
        return samples.reshape((-1, self.N))

    def probability(self, sample: ArrayLike) -> float:
        if self.N >= 15:
            raise ValueError("Probability computation for spins > 15 is not supported")

        return (
            jnp.exp(-energy(sample, self.J, self.b, self.J_sparse) / self.T)
            / self.Z
        )


def generate_isometric_ising(
    n_samples: int = 100, T: float = 2.5, grid_size: int = 4
) -> (ndarray, None):
    """Generate data for an Ising model with a 2D grid.
    """
    num_spins = grid_size * grid_size
    num_chains = 2
    num_steps = 1000
    J = create_2D_interaction_matrix(grid_size)
    model = IsingSpins(num_spins, J, b=1.0, T=T)
    # Plot the magnetization and energy trajectories for a single T
    samples = model.sample(
        n_samples * num_steps, num_chains=num_chains, num_warmup=10000, key=0
    )
    return samples[-n_samples:], None


if __name__ == "__main__":
    numpyro.set_host_device_count(2)
    samples, _ = generate_isometric_ising()
    print(samples.shape)
