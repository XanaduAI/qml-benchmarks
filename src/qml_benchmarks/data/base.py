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

"""Base classes for data generators."""

from abc import ABC, abstractmethod
from jax.typing import ArrayLike


class SpinGeneratorBase(ABC):
    """A base class for generating spin states according to some distribution.
    """
    def __init__(self, N: int) -> None:
        """
        A base class for generating spin states according to some distribution.

        Args:
            N (int): The number of spins.
        """
        self.N: int = N

    @abstractmethod
    def probability(self, sample: ArrayLike) -> float:
        """
        Computes the probability of a given spin configuration, if possible.

        Args:
            sample: A specific sample configuration of spins.

        Returns:
            float: The probability for the given configuration.
        """

    @abstractmethod
    def sample(self, n_samples: int) -> ArrayLike:
        """
        Samples states from the defined probability distribution.

        Args:
            n_samples (int): The number of states to sample.

        Returns:
            np.ndarray: An array of sampled spin states.
        """
