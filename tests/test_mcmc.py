"""Tests for the MCMC implementation using JAX and Numpyro."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from numpyro.infer import MCMC
from qml_benchmarks.mcmc import MetropolisHastings


# Simple energy functions for which we should know the posteriors for testing
@jax.jit
def energy_sum(x: jnp.array) -> float:
    """Simple energy function as the sum of the spins.

    Args:
        x (jnp.array): State of the system.

    Returns:
        float: Energy value.
    """
    return jnp.sum(x)


@pytest.mark.parametrize("num_samples", [10, 100, 1000])
@pytest.mark.parametrize("num_chains", [2, 4])
@pytest.mark.parametrize("dim", [1, 3, 5])
def test_mcmc_runs(num_samples, num_chains, dim):
    """Testing that the MCMC implementation runs."""

    kernel = MetropolisHastings(energy_sum)

    # Create initial state with random values in [-1, 1]
    key = random.PRNGKey(0)
    init_params = random.choice(key, jnp.array([-1, 1]), shape=(num_chains, dim))

    mcmc = MCMC(kernel, num_warmup=1000, num_samples=num_samples, num_chains=num_chains)

    mcmc.run(random.PRNGKey(0), init_params=init_params)
    posterior_samples = mcmc.get_samples()
    assert posterior_samples.shape == (num_samples * num_chains, dim)
    assert jnp.all(jnp.isin(posterior_samples, jnp.array([1, -1])))


@pytest.mark.parametrize(
    "energy_fn",
    [
        energy_sum,
    ],
)
@pytest.mark.parametrize("num_samples", [10000])
@pytest.mark.parametrize("dim", [3, 4, 5])
def test_mcmc(energy_fn, num_samples, dim):
    """Test MCMC sampling with different energy functions"""

    # Initialize the kernel with the potential function
    kernel = MetropolisHastings(energy_fn)
    num_chains = 4

    # Create initial state with random values in [-1, 1]
    key = random.PRNGKey(0)
    init_params = random.choice(key, jnp.array([-1, 1]), shape=(num_chains, dim))

    # Run the MCMC
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=num_samples, num_chains=num_chains)
    mcmc.run(random.PRNGKey(0), init_params=init_params)

    samples = mcmc.get_samples()
    assert jnp.all(jnp.isin(samples, jnp.array([1, -1])))

    energies = jax.vmap(energy_fn)(samples)
    energy_values, counts = np.unique(np.array(energies), return_counts=True)
    energy_hist = dict(zip(tuple(counts), tuple(energy_values)))

    # Check that the lowest energy value appears most frequently
    assert energy_hist[np.max(counts)] == jnp.min(energies)

    # Check that the highest energy value appears least frequently
    assert energy_hist[np.min(counts)] == jnp.max(energies)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
