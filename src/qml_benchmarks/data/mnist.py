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

"""
Note: Requires the torch, torchvision and keras packages to be installed to download the original data.
Since these are large they are not listed as dependencies of this repo.

The return type differs from other data generators since we want to reproduce the original
MNIST train/test split.
"""

import torchvision
import torchvision.transforms as transforms
from keras.datasets import mnist
from random import choices
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def generate_mnist(
    digitA, digitB, preprocessing, n_features=None, n_samples=None, height=None
):
    if preprocessing == "cg":

        mnist_train = torchvision.datasets.MNIST(
            "mnist_original/", download=True, train=True
        )
        X_train = mnist_train.data
        y_train = mnist_train.targets

        mnist_test = torchvision.datasets.MNIST(
            "mnist_original/", download=True, train=False
        )
        X_test = mnist_test.data
        y_test = mnist_test.targets

        idxs_train = np.concatenate(
            (np.where(y_train == digitA)[0], np.where(y_train == digitB)[0])
        )
        idxs_test = np.concatenate(
            (np.where(y_test == digitA)[0], np.where(y_test == digitB)[0])
        )

        X_train = X_train[idxs_train]
        y_train = y_train[idxs_train]
        X_test = X_test[idxs_test]
        y_test = y_test[idxs_test]

        # make \pm1 labels
        y_train[np.where(y_train == digitA)] = -1
        y_train[np.where(y_train == digitB)] = 1
        y_test[np.where(y_test == digitA)] = -1
        y_test[np.where(y_test == digitB)] = 1

        X_train = np.reshape(X_train, (X_train.shape[0], 1, 28, 28))
        X_train = transforms.Resize((height, height))(X_train)
        X_test = np.reshape(X_test, (X_test.shape[0], 1, 28, 28))
        X_test = transforms.Resize((height, height))(X_test)

        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        # scale the data
        scaler = StandardScaler()
        X_train_flat = scaler.fit_transform(X_train_flat)
        X_test_flat = scaler.transform(X_test_flat)

        return X_train_flat, X_test_flat, y_train, y_test

    if preprocessing in ["pca", "pca-"]:

        (X_train_original, y_train), (X_test_original, y_test) = mnist.load_data()

        # subselect a binary classification task
        pick_train = [i for i, y in enumerate(y_train) if (y == digitA or y == digitB)]
        X_train_original = X_train_original[pick_train]
        y_train = y_train[pick_train]
        y_train = np.array([-1.0 if y == digitA else 1.0 for y in y_train])

        pick_test = [i for i, y in enumerate(y_test) if (y == digitA or y == digitB)]
        X_test_original = X_test_original[pick_test]
        y_test = y_test[pick_test]
        y_test = np.array([-1.0 if y == digitA else 1.0 for y in y_test])

        # final number of features (between 1 and 28x28)
        train_X_flat = np.reshape(X_train_original, (-1, 28 * 28))
        test_X_flat = np.reshape(X_test_original, (-1, 28 * 28))

        scaler = StandardScaler()
        train_X_flat = scaler.fit_transform(train_X_flat)
        # note: we apply the scaler from the training set for best practice,
        # because in real life we might not have the test set at train time
        test_X_flat = scaler.transform(test_X_flat)

        pca = PCA(n_components=n_features)
        X_train = pca.fit_transform(train_X_flat)
        # note: we apply pca from the training set for best practice,
        # because in real life we might not have the test set at train time
        X_test = pca.transform(test_X_flat)

        if preprocessing == "pca-":
            # subsample smaller number of data
            rnd_indices_train = choices(list(range(len(X_train))), k=n_samples)
            X_train = X_train[rnd_indices_train]
            y_train = y_train[rnd_indices_train]

            rnd_indices_test = choices(list(range(len(X_test))), k=n_samples)
            X_test = X_test[rnd_indices_test]
            y_test = y_test[rnd_indices_test]

        return X_train, X_test, y_train, y_test
