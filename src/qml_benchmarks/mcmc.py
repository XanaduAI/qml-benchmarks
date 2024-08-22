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

"""Markov-chain Monte Carlo (MCMC) implementation with JAX and Numpyro."""

from collections import namedtuple
from functools import partial

import jax.numpy as jnp
import numpyro.distributions as dist
from jax import random
from numpyro.infer.mcmc import MCMCKernel

MHState = namedtuple("MHState", ["state", "rng_key"])


class MetropolisHastings(MCMCKernel):
    """A simple Metropolis-Hastings MCMC kernel.

    Args:
       potential_fn (callable):
           A callable representing the energy function.
    """

    sample_field = "state"

    def __init__(self, potential_fn):
        """_summary_

        Args:
            potential_fn (callable): Potenital energy function.
        """
        self.potential_fn = potential_fn

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        """_summary_

        Args:
            rng_key (_type_): _description_
            num_warmup (_type_): _description_
            init_params (_type_): _description_
            model_args (_type_): _description_
            model_kwargs (_type_): _description_

        Returns:
            _type_: _description_
        """
        return MHState(init_params, rng_key)

    def sample(self, state, model_args, model_kwargs):
        """_summary_

        Args:
            state (_type_): _description_
            model_args (_type_): _description_
            model_kwargs (_type_): _description_

        Returns:
            _type_: _description_
        """
        u, rng_key = state
        rng_key, key_proposal, key_accept = random.split(rng_key, 3)
        u_proposal = self.proposal(key_proposal, u)

        accept_prob = jnp.exp(
            self.potential_fn(u, *model_args, **model_kwargs)
            - self.potential_fn(u_proposal, *model_args, **model_kwargs)
        )
        u_new = jnp.where(
            dist.Uniform().sample(key_accept) < accept_prob, u_proposal, u
        )
        return MHState(u_new, rng_key)
        # spins, rng_key = state
        # num_spins = spins.size

        # def mh_step(i, val):
        #     spins, rng_key = val
        #     rng_key, subkey = random.split(rng_key)
        #     flip_index = random.randint(subkey, (), 0, num_spins)
        #     spins_proposal = spins.at[flip_index].set(-spins[flip_index])

        #     current_energy = self.potential_fn(
        #         spins, *model_args, **model_kwargs)
        #     proposed_energy = self.potential_fn(
        #         spins_proposal, *model_args, **model_kwargs)
        #     delta_energy = proposed_energy - current_energy
        #     accept_prob = jnp.exp(-delta_energy)

        #     rng_key, subkey = random.split(rng_key)
        #     accept = random.uniform(subkey) < accept_prob
        #     spins = jnp.where(accept, spins_proposal, spins)
        #     return spins, rng_key

        # spins, rng_key = jax.lax.fori_loop(0, num_spins, mh_step, (spins, rng_key))
        # return MHState(spins, rng_key)

    def proposal(self, key, u):
        """Make a new proposal by flipping spins randomly.

        Args:
            key (_type_): _description_
            u (_type_): _description_

        Returns:
            _type_: _description_
        """
        num_spins = u.size
        # Generate a number of indices for flipping spins
        flip_indices = random.randint(key, (num_spins,), 0, num_spins)
        # Flip the selected spins
        u_proposal = u.at[flip_indices].set(-u[flip_indices])
        return u_proposal
