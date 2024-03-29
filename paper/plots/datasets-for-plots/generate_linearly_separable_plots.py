"""Generates linearly separable data in any dimension.

The data is uniformly sampled from insde a hypercube, and the labels
generated by a linear teacher model with weight vector [1,...,1] and zero
offset.
"""
import numpy as np
from sklearn.model_selection import train_test_split
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# set the random seed
seed = 42
np.random.seed(seed)

# make the folders with the name of the dataset
folder_name = 'linearly_separable'
os.makedirs(folder_name, exist_ok=True)

n_features =2

n_samples = 300
margin = 0.02*n_features

w_true = np.ones(n_features)

# sample more data than we need randomly from a hypercube
X = 2*np.random.rand(2*n_samples, n_features)-1

# only retain data outside a margin
X = [x for x in X if np.abs(np.dot(x, w_true)) > margin]
X = X[:n_samples]

y = [np.dot(x, w_true) for x in X]
y = [-1 if y_ > 0 else 1 for y_ in y]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

name = f"linearly_separable_{n_features}d"
np.savetxt(os.path.join(folder_name, name + '_train.csv'), np.c_[X_train, y_train], delimiter=',')
np.savetxt(os.path.join(folder_name, name + '_test.csv'), np.c_[X_test, y_test], delimiter=',')


