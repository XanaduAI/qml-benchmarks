# Benchmarking for quantum machine learning models

This repository contains tools to compare the performance of near-term quantum machine learning (QML)
as well as standard classical machine learning models on supervised and generative learning tasks. 

It is based on pipelines using [Pennylane](https://pennylane.ai/) for the simulation of quantum circuits, 
[JAX](https://jax.readthedocs.io/en/latest/index.html) for training, 
and [scikit-learn](https://scikit-learn.org/) for the benchmarking workflows. 

Version 0.1 of the code can be used to reproduce the results in the study 
"Better than classical? The subtle art of benchmarking quantum machine learning models".

## Overview

A short summary of the various folders in this repository is as follows:
- `paper`: contains code and results to reproduce the results in the paper
  - `benchmarks`: scripts that generate datasets of varied difficulty and feature dimensions
  - `plots`: scripts that generate the plots and additional experiments in the paper
  - `results`: data files recording the results of the benchmark experiments that the study is based on
- `scripts`: example code for how to benchmark a model on a dataset
- `src/qml_benchmarks`: a simple Python package defining quantum and classical models, 
   as well as data generating functions

## Installation

You can install the `qml_benchmarks` package in your environment with

```bash
pip install -e .
```

from the root directory of the repository. This will install the package in
editable mode, meaning that changes to the code will be reflected in the
installed package.

Dependencies of this package can be installed in your environment by running 

```bash
pip install -r requirements.txt
```

## Adding a custom classifier

We use the [Scikit-learn API](https://scikit-learn.org/stable/developers/develop.html) to create 
models and perform hyperparameter search.

A minimal template for a new quantum classifier is as follows, and can be stored 
in `qml_benchmarks/models/my_model.py`:

```python
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin


class MyModel(BaseEstimator, ClassifierMixin):
    def __init__(self, hyperparam1="some_value",  random_state=42):

        # store hyperparameters as attributes
        self.hyperparam1 = hyperparam1
                    
        # reproducibility is ensured by creating a numpy PRNG and using it for all
        # subsequent random functions. 
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
            
        # define data-dependent attributes
        self.params_ = None
        self.n_qubits_ = None
        
    def initialize(self, args):
        """
        initialize the model if necessary
        """
        # ... your code here ...   

    def fit(self, X, y):
        """Fit the model to data X and labels y.

        Add your custom training loop here and store the trained model parameters in `self.params_`.
        
        Args:
            X (array_like): Data of shape (n_samples, n_features)
            y (array_like): Labels of shape (n_samples,)
        """
        # ... your code here ...        
        

    def predict(self, X):
        """Predict labels for data X.
        
        Args:
            X (array_like): Data of shape (n_samples, n_features)
        
        Returns:
            array_like: Predicted labels of shape (n_samples,)
        """
        # ... your code here ...
        
        return y_pred

    def predict_proba(self, X):
        """Predict label probabilities for data X.

        Args:
            X (array_like): Data of shape (n_samples, n_features)

        Returns:
            array_like: Predicted label probabilities of shape
            (n_samples, n_classes)
        """
        # ... your code here ...
        return y_pred_proba
```

To ensure compatibility with scikit-learn functionalities, all models should
inherit the `BaseEstimator` and `ClassifierMixin` classes.  Implementing the `fit`,
`predict`, and `predict_proba` methods is sufficient. 

The model parameters are stored as a dictionary in `self.params_`. 

There are two types of other attributes: those initialized when the instance of the class is 
created, and those that are only known when data is seen (for example, the number of qubits 
may depend on the dimension of input vectors). In the latter case, a default (i.e., `self.n_qubits_ = None`) 
is set in the `init` function, and the value is typically updated when `fit` is called for the first time.

It can be useful to implement an `initialize` method which initializes an untrained model with random 
parameters so that `predict_proba` and `predict` can be called. 

The custom model can be used as follows:

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from qml_benchmarks.models.my_model import MyModel

# load data and use labels -1, 1
X, y = make_classification(n_samples=100, n_features=2,
                           n_informative=2, n_redundant=0, random_state=42)
y = np.array([-1 if y_ == 0 else 1 for y_ in y])

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# fit model
model = MyModel(hyperparam1=0.5)
model.fit(X_train, y_train)

# score the model
print(model.score(X_test, y_test))
```


## Adding a custom generative model

The minimal template for a new generative model closely follows that of the classifier models.
Labels are set to `None` throughout to maintain sci-kit learn functionality. 

```python
import numpy as np

from sklearn.base import BaseEstimator


class MyModel(BaseEstimator):
    def __init__(self, hyperparam1="some_value",  random_state=42):

        # store hyperparameters as attributes
        self.hyperparam1 = hyperparam1
                    
        # reproducibility is ensured by creating a numpy PRNG and using it for all
        # subsequent random functions. 
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
            
        # define data-dependent attributes
        self.params_ = None
        self.n_qubits_ = None
        
    def initialize(self, args):
        """
        initialize the model if necessary
        """
        # ... your code here ...   

    def fit(self, X, y=None):
        """Fit the model to data X.

        Add your custom training loop here and store the trained model parameters in `self.params_`.
        
        Args:
            X (array_like): Data of shape (n_samples, n_features)
            y (array_like): not used (no labels)
        """
        # ... your code here ...        

    def sample(self, num_samples):
        """sample from the generative model
        
        Args:
            num_samples (int): number of points to sample
        
        Returns:
            array_like: sampled points
        """
        # ... your code here ...
        
        return samples

    def score(self, X, y=None):
        """A optional custom score function to be used with hyperparameter optimization
        Args:
            X (array_like): Data of shape (n_samples, n_features)
            y: unused (no labels for generative models)

        Returns:
            (float): score for the dataset X
        """
        # ... your code here ...
        return score
```

If the model samples binary data, it is recommended to construct models that sample binary strings (rather than $\pm1$ valued strings) 
to align with the datasets designed for generative models. The repository currently contains two classical generative models:
a restricted Boltzmann machine and a simple energy based model (called DeepEBM) that uses a multi-layer perceptron as its energy function. 
Energy based models with more structure can easily be constructed by replacing the multilayer perception neural network by
any other differentiable network written in flax. 

## Datasets

The `qml_benchmarks.data` module provides generating functions to create datasets for binary classification and
generative learning.

A generating function can be used like this:

```python
from qml_benchmarks.data import generate_two_curves

X, y = generate_two_curves(n_samples=200, n_features=4, degree=3, noise=0.1, offset=0.01)
```

Note that some datasets might have different return data structures, for example if the train/test split 
is performed by the generating function. If the dataset does not include labels, `y = None` is returned. 

The original datasets used in the paper can be generated by running the scripts in the `paper/benchmarks` folder, 
such as:

```bash
python paper/benchmarks/generate_hyperplanes.py
```

This will create a new folder in `paper/benchmarks` containing the datasets.

## Running hyperparameter optimization

In the folder `scripts` we provide an example that can be used to
generate results for a hyperparameter search for any model and dataset. The script functions
for both classifier and generative models. The script
can be run as

```
python run_hyperparameter_search.py --model "DataReuploadingClassifier" --dataset-path "my_dataset.csv"
```

where`my_dataset.csv` is a CSV file containing the training data. For classification problems, each column should 
correspond to a feature and the last column to the target. For generative learning, each row
should correspond to a binary string that specifies a unique data sample, and the model should implement a `score`
method. 

Unless otherwise specified, the hyperparameter grid is loaded from `qml_benchmarks/hyperparameter_settings.py`.
One can override the default grid of hyperparameters by specifying the hyperparameter list,
where the datatype is inferred from the default values.
For example, for the `DataReuploadingClassifier` we can run:

```
python run_hyperparameter_search.py \
    --model DataReuploadingClassifier \
    --dataset-path "my_dataset.csv" \
    --n_layers 1 2 \
    --observable_type "single" "full"\
    --learning_rate 0.001 \
    --clean True
```

which runs a search for the grid:

```
{'max_vmap': array([1]), 
'batch_size': array([32]), 
'learning_rate': [0.001]), 
'n_layers': [1, 2], 
'observable_type': ['single']}
```

The script creates two CSV files that contains the detailed results of hyperparameter search and the best 
hyperparameters obtained in the search. These files are similar to the ones stored in the `paper/results`
folder. 

The best hyperparameters can be loaded into a model and used to score the classifier.

You can check the various options for the script using:

```
python run_hyperparameter_search --help
```

## Feedback 

Please help us improve this repository and report problems by opening an issue or pull request.
