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

"""Utility functions for hyperparameter search"""

import csv
import numpy as np
import pandas as pd


def read_data(path, labels=True):
    """Read data from a csv file where each row is a data sample.
    The columns are the input features and the last column specifies a label.

    Return a 2-d array of inputs and an array of labels, X,y.

    Args:
        path (str): path to data
    """
    # The data is stored on a CSV file with the last column being the label
    data = pd.read_csv(path, header=None)
    if labels:
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
    else:
        X = data.iloc[:, :].values
        y = None
    return X, y


def construct_hyperparameter_grid(hyperparameter_settings, classifier_name):
    """Constructs a grid of hyperparameters from the dictionary of hyperparameter
    settings for a given classifier.

    Args:
        hyperparameter_settings (dict): a dictionary of hyperparameter settings
        classifier_name (str): classifier name

    Returns:
        hyperparameter_grid (dict): A grid of hyperparameters to search
    """
    hyperparams = hyperparameter_settings[classifier_name].keys()
    hyperparameter_grid = {}

    for hyperparam in hyperparams:
        if hyperparameter_settings[classifier_name][hyperparam]["type"] == "list":
            val = hyperparameter_settings[classifier_name][hyperparam]["val"]
            dtype = hyperparameter_settings[classifier_name][hyperparam]["dtype"]
            if dtype == "tuple":
                hyperparameter_grid[hyperparam] = [eval(v) for v in val]
            else:
                hyperparameter_grid[hyperparam] = np.array(val, dtype=dtype)

    return hyperparameter_grid


def csv_to_dict(file_path):
    """Read a csv file and interpret the content as a dictionary.

    Args:
        file_path (str): path to csv file
    """
    dict = {}
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        # Skip the first line
        next(csvreader)
        for row in csvreader:
            hyperparameter, value = row
            # Check if the value is numeric and convert it to int or float accordingly
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass  # If conversion is not possible, keep the value as a string
            dict[hyperparameter] = value
    return dict
