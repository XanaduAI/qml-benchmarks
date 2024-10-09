"""Ising spin simulation for a classical 2D Ising model
"""

import jax
from jax import numpy as jnp
from numpy import ndarray
from numpyro.infer import MCMC
from jax import random
from collections import namedtuple
from numpyro.infer.mcmc import MCMCKernel
from qgml.data import SpinConfigurationGeneratorBase
from tqdm.auto import tqdm

def create_isotropic_interaction_matrix(grid_size: int):
    """Create an interaction matrix for a 2D isotropic square lattice."""
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
def energy(s, J, b, J_sparse=None):
    """Calculate the Ising energy. For sparse Hamiltonians, it is recommneded to supply a list of nonzero indices of
    J to speed up the calculation.
    Args:
        s: spin configuration
        J: interaction matrix
        b: bias term
        J_sparse: list of nonzero indices of J.
    """
    if J_sparse is not None:
        return -jnp.einsum(
            "i,i,i->", s[J_sparse[0]], s[J_sparse[1]], J[J_sparse]
        ) / 2.0 - jnp.dot(s, b)
    else:
        return -jnp.einsum("i,j,ij->", s, s, J) / 2.0 - jnp.dot(s, b)


def initialize_spins(rng_key, num_spins, num_chains):
    if num_chains == 1:
        spins = random.bernoulli(rng_key, 0.5, (num_spins,))
    else:
        spins = random.bernoulli(
            rng_key,
            0.5,
            (
                num_chains,
                num_spins,
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
        """Sample from the model via Metropolis Hastings MCMC"""
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


# Define the Ising model class
class IsingSpins(SpinConfigurationGeneratorBase):
    """
    class object used to generate datasets
    ArgsL
    N (int): Number of spins
    J (np.array): interaction matrix
    b (np.array): bias terms
    T (float): temperature
    sparse (bool): If true, J is converted to a sparse representation (faster for sparse Hamiltonians)
    compute_partition_fn: Whether to compute the partition function
    """
    def __init__(
        self, N: int, J: jnp.array, b: jnp.array, T: float, sparse=False, compute_partition_fn=False
    ) -> None:
        super().__init__(N)
        self.kernel = MetropolisHastings()
        self.J = J
        self.T = T
        self.b = b
        self.J_sparse = jnp.nonzero(J) if sparse else None

       if compute_partition_fn:
            Z = 0
            for i in tqdm(range(2**self.N), desc="Computing partition function"):
                lattice = (-1) ** jnp.array(jnp.unravel_index(i, [2] * self.N))
                en = energy(lattice, self.J, self.b, self.J_sparse)
                Z += jnp.exp(-en / T)
            self.Z = Z

    def sample(
        self, num_samples: int, num_chains=1, thinning=1, num_warmup=1000, key=42
    ) -> jnp.array:

        """
        Generate samples.
        Args:
            num_samples (int): total number of samples to generate per chain
            num_chains (int): number of chains
            thinning (int): how much to thin the sampling. e.g. if thinning = 10 a sample will be drawn ater each
                10 steps of mcmc sampling. Larger numbers result in more unbiased samples.
            num_warmup (int): number of mcmc 'burn in' steps to perform before collecting any samples.
            key (int): random seed used to initialize sampling.
        """
        rng_key = random.PRNGKey(key)
        init_spins = initialize_spins(rng_key, self.N, num_chains)
        mcmc = MCMC(
            self.kernel,
            num_warmup=num_warmup,
            thinning=thinning,
            num_samples=num_samples,
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

    def probability(self, spin_configuration: ndarray) -> float:
        return (
            jnp.exp(-energy(spin_configuration, self.J, self.b, self.J_sparse) / self.T)
            / self.Z
        )

def generate_isometric_ising(
    num_samples: int = 100, T: float = 2.5, grid_size: int = 4
) -> (ndarray, None):
    num_spins = grid_size * grid_size
    num_chains = 2
    num_steps = 1000
    J = create_isotropic_interaction_matrix(grid_size)
    model = IsingSpins(num_spins, J, b=1.0, T=T)
    # Plot the magnetization and energy trajectories for a single T
    samples = model.sample(num_samples*num_steps, num_chains=num_chains, num_warmup=10000, key=0)
    return samples[-num_samples:], None
