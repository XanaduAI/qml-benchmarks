"""Ising spin simulation for a classical 2D Ising model
"""

import jax
from jax import numpy as jnp
from numpy import ndarray
from numpyro.infer import MCMC
from jax import random
from collections import namedtuple
from numpyro.infer.mcmc import MCMCKernel
from tqdm.auto import tqdm


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


class IsingSpins:
    r"""
    class object used to generate datasets by sampling an ising distrbution of a specified interaction
    matrix. The distribution is sampled via markov chain Monte Carlo via the Metrolopis Hastings
    algorithm.

    In the case of perfect sampling, a spin configuration s is sampled with probabability
    :math:`p(s)=exp(-H(s)/T)`, where the energy :math:`H(s)=\sum_{i\neq j}s_i s_i J_{ij}+\sum_i b_i s_i`
    corresponds to an ising Hamiltonian and configurations s are :math:`\pm1` valued.

    The final sampled configurations are converted from a :math:`\pm1` representation to to a binary
    representation via x = (s+1)//2.

    N (int): Number of spins
    J (np.array): interaction matrix
    b (np.array): bias terms
    T (float): temperature
    sparse (bool): If true, J is converted to a sparse representation (faster for sparse Hamiltonians)
    compute_partition_fn: Whether to compute the partition function
    """

    def __init__(
        self,
        N: int,
        J: jnp.array,
        b: jnp.array,
        T: float,
        sparse=False,
        compute_partition_fn=False,
    ) -> None:
        self.N = N
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
        samples.reshape((-1, self.N))
        return (samples + 1) // 2

    def probability(self, x: ndarray) -> float:
        """
        compute the probability of a binary configuration x
        Args:
            x: binary configuration array
        Returns:
            (float): the probability of sampling x according to the ising distribution
        """

        if not (hasattr(self, "Z")):
            raise Exception(
                "probability requires partition function to have been computed"
            )

        return jnp.exp(-energy(2*x-1, self.J, self.b, self.J_sparse) / self.T) / self.Z


def generate_ising(
    N: int,
    num_samples: int,
    J: jnp.array,
    b: jnp.array,
    T: float,
    sparse=False,
    num_chains=1,
    thinning=1,
    num_warmup=1000,
    key=42,
):
    r"""
    Generating function for Ising datasets.

    The dataset is generated by sampling an ising distrbution of a specified interaction
    matrix. The distribution is sampled via Markov Chain Monte Carlo via the Metrolopis Hastings
    algorithm.

    In the case of perfect sampling, a spin configuration s is sampled with probabability
    :math:`p(s)=exp(-H(s)/T)`, where the energy :math:`H(s)=\sum_{i\neq j}s_i s_i J_{ij}+\sum_i b_i s_i`
    corresponds to an ising Hamiltonian and configurations s are :math:`\pm1` valued.

    The final sampled configurations are converted from a :math:`\pm1` representation to to a binary
    representation via x = (s+1)//2.

    Note that in order to use parallelization, the number of avaliable cores has to be specified explicitly
    to numpyro. i.e. the line `numpyro.set_host_device_count(num_cores)` should appear before running the
    generator, where num_cores is the number of avaliable CPU cores you want to use.

    N (int): Number of spins
    num_samples (int): total number of samples to generate per chain
    J (np.array): interaction matrix of shape (N,N)
    b (np.array): bias array of shape (N,)
    T (float): temperature
    num_chains (int): number of chains, defaults to 1.
    thinning (int): how much to thin the sampling. e.g. if thinning = 10 a sample will be drawn after each
        10 steps of mcmc sampling. Larger numbers result in more unbiased samples.
    num_warmup (int): number of mcmc 'burn in' steps to perform before collecting any samples.
    key (int): random seed used to initialize sampling.
    sparse (bool): If true, J is converted to a sparse representation (faster for sparse Hamiltonians)

    Returns:
        Array of data samples, and Nonetype object (since there are no labels)
    """

    sampler = IsingSpins(N, J, b, T, sparse=sparse, compute_partition_fn=False)
    samples = sampler.sample(
        num_samples,
        num_chains=num_chains,
        thinning=thinning,
        num_warmup=num_warmup,
        key=key,
    )
    return samples, None
