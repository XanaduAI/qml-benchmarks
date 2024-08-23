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

import flax.linen as nn
from qml_benchmarks.models.base import EnergyBasedModel, BaseGenerator
from sklearn.neural_network import BernoulliRBM
from joblib import Parallel, delayed
from qml_benchmarks.model_utils import mmd_loss, median_heuristic
import numpy as np


class MLP(nn.Module):
    "Multilayer perceptron."
    # Create a MLP with hidden layers and neurons specfied as a list of integers.
    hidden_layers: list[int]

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_layers:
            x = nn.Dense(dim)(x)
            x = nn.tanh(x)
        x = nn.Dense(1)(x)
        return x


class DeepEBM(EnergyBasedModel):
    """
    Energy-based model with the energy function is a neural network.

    Args:
        hidden_layers (list[int]):
            The number of hidden layers and neurons in the MLP layers.
    """

    def __init__(self, hidden_layers=[8, 4],
                 mmd_kwargs = {'n_samples': 1000, 'n_steps':1000, 'sigma': 1.0},
                 **base_kwargs):
        super().__init__(**base_kwargs)
        self.hidden_layers = hidden_layers
        self.model = MLP(hidden_layers=hidden_layers)
        self.mmd_kwargs = mmd_kwargs

    def initialize(self, x):
        dim = x.shape[1]
        if not isinstance(dim, int):
            raise NotImplementedError(
                "The model is not yet implemented for data"
                "with arbitrary dimensions. `dim` must be an integer."
            )

        self.dim = dim
        self.params_ = self.model.init(self.generate_key(), x)

    def energy(self, params, x):
        return self.model.apply(params, x)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        sigma = self.mmd_kwargs['sigma']
        sigmas = [sigma] if isinstance(sigma, (int, float)) else sigma
        score = np.mean([mmd_loss(X, self.sample(self.mmd_kwargs['n_samples'],
                                                 self.mmd_kwargs['n_steps']), sigma) for sigma in sigmas])
        return float(-score)


class RestrictedBoltzmannMachine(BernoulliRBM, BaseGenerator):
    def __init__(
        self,
        n_components=256,
        learning_rate=0.1,
        batch_size=10,
        n_iter=10,
        verbose=0,
        random_state=None,
        score_fn='pseudolikelihood',
        mmd_kwargs ={'n_samples': 1000, 'n_steps': 1000, 'sigma': 1.0}
    ):
        super().__init__(
            n_components=n_components,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_iter=n_iter,
            verbose=verbose,
            random_state=random_state,
        )
        self.score_fn = score_fn
        self.mmd_kwargs = mmd_kwargs

    def initialize(self, X: any = None):
        if len(X.shape) > 2:
            raise ValueError("Input data must be 2D")
        self.dim = X.shape[1]

    def fit(self, X, y=None):
        self.initialize(X)
        super().fit(X, y)

    # Gibbs sampling:
    def _sample(self, num_steps=1000):
        """
        Sample the model for given number of steps.

        Args:
            num_steps (int): Number of Gibbs sample steps

        Returns:
            np.array: The samples at the given temperature.
        """
        if self.dim is None:
            raise ValueError("Model must be initialized before sampling")
        v = np.random.choice(
            [0, 1], size=(self.dim,)
        )  # Assuming `N` is `self.n_components`
        for _ in range(num_steps):
            v = self.gibbs(v)  # Assuming `gibbs` is an instance method
        return v

    def sample(self, num_samples: int, num_steps: int = 1000) -> np.ndarray:
        # Parallelize the sampling process
        samples_t = Parallel(n_jobs=-1)(
            delayed(self._sample)(num_steps=num_steps) for _ in range(num_samples)
        )
        samples_t = np.array(samples_t, dtype=int)
        return samples_t

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        if self.score_fn == 'pseudolikelihood':
            return float(np.mean(super().score_samples(X)))
        elif self.score_fn == 'mmd':
            sigma = self.mmd_kwargs['sigma']
            sigmas = [sigma] if isinstance(sigma, (int, float)) else sigma
            score = np.mean([mmd_loss(X, self.sample(self.mmd_kwargs['n_samples'],
                                                     self.mmd_kwargs['n_steps']), sigma) for sigma in sigmas])
            return float(-score)