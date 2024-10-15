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

"""Generate a dataset of spin configurations as blobs."""

import numpy as np


class RandomSpinBlobs:
    """
    Class object used to generate spin blob datasets: a binary analog of the
    'gaussian blobs' dataset, in which bitstrings are sampled close in Hamming
    distance to a set of specified configurations.

    The dataset is generated by specifying a list of configurations (peak spins)
    that mark the centre of the 'blobs'. Data points are sampled by chosing one of
    the peak spins (with probabilities specified by peak probabilities), and then
    flipping some of the bits. Each bit is flipped with probability specified by
    p, so that (for small p) datapoints are close in Hamming distance to one of
    the peak probabilities.

    Args:
        N (int): The number of spins.
        num_blobs (int):
            The number of blobs.
        peak_probabilities (list[float], optional):
            The probability of each spin to be selected. If not specified,
            the probabilities are distributed uniformly.
        peak_spins (list[np.array], optional):
            The peak spin configurations. Selected randomly by default.
        p (float, optional):
            The value of the parameter `p` in a Binomial distribution specifying
            the number of spins that are flipped each time during sampling.
            Defaults to 0.01.
    """

    def __init__(
        self,
        N: int,
        num_blobs: int,
        peak_probabilities: list[float] = None,
        peak_spins: list[np.array] = None,
        p: float = 0.01,
    ) -> None:
        self.N = N
        self.num_blobs = num_blobs

        if peak_probabilities is None:
            peak_probabilities = [1 / num_blobs for i in range(num_blobs)]

        if len(peak_probabilities) != num_blobs:
            msg = f"Specify probabilities for all {num_blobs} blobs."
            raise ValueError(msg)

        if peak_spins is not None and len(peak_spins) != num_blobs:
            msg = f"The number of peak spins should be the same as blobs."
            raise ValueError(msg)

        if peak_spins is None:
            # generate some random peak spin configs
            # we flip each bit with 50% prob
            spin_configs = []
            while len(spin_configs) < num_blobs:
                config = list((-1) ** np.random.binomial([1] * N, [0.5] * N))
                if config not in spin_configs:
                    spin_configs.append(config)
            peak_spins = spin_configs

        self.peak_spins = peak_spins
        self.peak_probabilities = peak_probabilities
        self.p = p

    def sample(self, num_samples: int, return_labels=False) -> np.array:
        """Generate a given number of samples.

        Args:
            num_samples (int): Number of samples to generate.
            return_labels (bool, optional):
                Whether to return labels for each sample. Defaults to False.

        Returns:
            np.array: A (num_samples, N) array of spin configurations.
        """
        samples = []
        labels = []

        for _ in range(num_samples):
            # Choose a random peak
            label = np.random.choice(self.num_blobs, p=self.peak_probabilities)
            labels.append(label)
            peak_spin = self.peak_spins[label]

            # Randomly choose a Hamming distance from the sampler
            num_bits_to_flip = np.random.binomial(self.N, self.p)

            # Flip bits randomly in the peak spin configuration
            indices_to_flip = np.random.choice(self.N, num_bits_to_flip, replace=False)
            sampled_spin = np.copy(peak_spin)
            for index in indices_to_flip:
                sampled_spin[index] *= -1

            samples.append(sampled_spin)

        samples = (np.array(samples) + 1) / 2

        if return_labels:
            return samples, np.array(labels)
        else:
            return samples


def generate_spin_blobs(
    N: int,
    num_blobs: int,
    num_samples: int,
    peak_probabilities: list[float] = None,
    peak_spins: list[np.array] = None,
    p: float = 0.01,
):
    """
    Generator function for spin blob datasets: a binary analog of the
    'gaussian blobs' dataset, in which bitstrings are sampled close in Hamming
    distance to a set of specified configurations.

    The dataset is generated by specifying a list of configurations (peak spins)
    that mark the centre of the 'blobs'. Data points are sampled by chosing one of
    the peak spins (with probabilities specified by peak probabilities), and then
    flipping some of the bits. Each bit is flipped with probability specified by
    p, so that (for small p) datapoints are close in Hamming distance to one of
    the peak probabilities.

    Args:
        N (int): The number of spins.
        num_blobs (int):
            The number of blobs.
        num_samples (int): The number of samples to generate.
        peak_probabilities (list[float], optional):
            The probability of each spin to be selected. If not specified,
            the probabilities are distributed uniformly.
        peak_spins (list[np.array], optional):
            The peak spin configurations. Selected randomly by default.
        p (float, optional):
            The value of the parameter `p` in a Binomial distribution specifying
            the number of spins that are flipped each time during sampling.
            Defaults to 0.01.

    Returns:
        tuple(np.ndarray): Dataset array and label array specifying the peak spin
            that was used to sample each datapoint.
    """

    sampler = RandomSpinBlobs(
        N=N,
        num_blobs=num_blobs,
        peak_probabilities=peak_probabilities,
        peak_spins=peak_spins,
        p=p,
    )

    X, y = sampler.sample(num_samples=num_samples, return_labels=True)
    X = X.reshape(-1, N)

    return X, y


def generate_8blobs(
    num_samples: int,
    p: float = 0.01,
):
    """Generate 4x4 spin samples with 8 selected high-probability configurations

    Example
    -------
        import matplotlib.pyplot as plt
        from qml_benchmarks.data.spin_blobs import generate_8blobs
        X, y = generate_8blobs(100)
        num_samples = 20
        interval = len(X) // num_samples

        fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
        for i in range(num_samples):
            axes[i].imshow(X[i*interval].reshape((4, 4)))
            axes[i].axis('off')
        plt.show()

    Args:
        num_samples (int): The number of samples to generate.
        p (float, optional):
            The value of the parameter p in a Binomial distribution bin(N, p)
            that determines how many spins are flipped during each sampling step
            after choosing one of the peak configurations. Defaults to 0.01.

    Returns:
        np.ndarray: A (num_samples, 16) array of spin configurations.
    """
    np.random.seed(66)
    N: int = 16
    num_blobs: int = 8

    # generate a specific set
    config1 = np.array(
        [[1, 1, -1, -1], [1, 1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]]
    )
    config2 = np.array(
        [[-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, -1, -1], [-1, -1, -1, -1]]
    )
    config3 = np.array(
        [[-1, -1, -1, -1], [-1, -1, -1, -1], [1, 1, -1, -1], [1, 1, -1, -1]]
    )
    config4 = np.array(
        [[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, 1, 1], [-1, -1, 1, 1]]
    )
    config5 = np.array(
        [[-1, -1, -1, -1], [-1, 1, 1, -1], [-1, 1, 1, -1], [-1, -1, -1, -1]]
    )
    config6 = np.array(
        [[1, -1, -1, -1], [-1, 1, -1, -1], [-1, -1, 1, -1], [-1, -1, -1, 1]]
    )
    config7 = np.array(
        [[-1, -1, -1, 1], [-1, -1, 1, -1], [-1, 1, -1, -1], [1, -1, -1, -1]]
    )
    config8 = np.array(
        [[1, -1, -1, 1], [-1, -1, -1, -1], [-1, -1, -1, -1], [1, -1, -1, 1]]
    )

    peak_spins = [
        np.reshape(config1, -1),
        np.reshape(config2, -1),
        np.reshape(config3, -1),
        np.reshape(config4, -1),
        np.reshape(config5, -1),
        np.reshape(config6, -1),
        np.reshape(config7, -1),
        np.reshape(config8, -1),
    ]
    sampler = RandomSpinBlobs(
        N=N,
        num_blobs=num_blobs,
        peak_spins=peak_spins,
        p=p,
    )

    X, y = sampler.sample(num_samples=num_samples, return_labels=True)
    X = X.reshape(-1, N)

    return X, y
