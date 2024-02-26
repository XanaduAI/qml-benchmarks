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

"""Score a model using the best hyperparameters, using a command-line script."""

import numpy as np
import sys
import os
import argparse
import logging
import pandas as pd
logging.getLogger().setLevel(logging.INFO)
from importlib import import_module
from pathlib import Path
from qml_benchmarks.hyperparam_search_utils import read_data, csv_to_dict

np.random.seed(42)

logging.info('cpu count:' + str(os.cpu_count()))


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run experiments with hyperparameter search.",
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--classifier-name",
        help="Classifier to run",
    )

    parser.add_argument(
        "--trainset-path",
        help="Path to the training set",
    )

    parser.add_argument(
        "--testset-path",
        help="Path to the test set",
    )

    parser.add_argument(
        "--hyperparams-path",
        default=".",
        help="Path to the file with the best hyperparameters",
    )

    parser.add_argument(
        "--results-path", default=".", help="Path to store the experiment results"
    )

    parser.add_argument(
        "--clean",
        help="True or False. Remove previous results if it exists",
        dest="clean",
        default=False,
        type=bool,
    )

    parser.add_argument(
        "--n-jobs", type=int, default=-1, help="Number of parallel threads to run"
    )

    # Parse the arguments along with any extra arguments that might be model specific
    args, unknown_args = parser.parse_known_args()

    if any(arg is None for arg in [args.classifier_name,
                                   args.trainset_path,
                                   args.testset_path]):
        msg = "\n================================================================================"
        msg += "\nA classifier from qml.benchmarks.model and dataset path are required. E.g., \n \n"
        msg += ("python score_with_best_hyperparameters \ \n"
                "--classifier DataReuploadingClassifier \ \n"
                "--trainset-path my_train_data.csv\n"
                "--testset-path my_test_data.csv\n")
        msg += "\nCheck all arguments for the script with \n"
        msg += "python score_with_best_hyperparameters --help\n"
        msg += "================================================================================"
        raise ValueError(msg)

    experiment_path = args.results_path
    results_path = os.path.join(experiment_path, "results")

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    ###################################################################
    # Get the classifier, dataset and best hyperparameters
    ###################################################################
    Classifier = getattr(
        import_module("qml_benchmarks.models"),
        args.classifier_name
    )
    classifier_name = Classifier.__name__

    # Load the data
    train_dataset_filename = os.path.join(args.trainset_path)
    X_train, y_train = read_data(train_dataset_filename)

    test_dataset_filename = os.path.join(args.testset_path)
    X_test, y_test = read_data(test_dataset_filename)

    # Construct output path
    dataset_path_obj = Path(args.trainset_path)
    results_filename_stem = " ".join(
            [Classifier.__name__ + "_" + dataset_path_obj.stem
             + "_GridSearchCV"])

    # If we have already run this experiment then continue
    path_out = os.path.join(results_path, results_filename_stem + "-best-hyperparams-results.csv")
    if os.path.isfile(path_out):
        if args.clean is False:
            msg = "\n================================================================================="
            msg += "\nResults exist in " + path_out
            msg += "\nSpecify --clean True to override results or new --results-path"
            msg += "\n================================================================================="
            logging.warning(msg)
            sys.exit(msg)
        else:
            logging.warning("Cleaning existing results for ", path_out)

    # Load best hyperparameters
    best_hyperparams = csv_to_dict(args.hyperparams_path)

    # Score the model
    results_with_best_hyperparams = {"train_acc": [], "test_acc": []}
    for i in range(5):
        classifier = Classifier(**best_hyperparams, random_state=i)
        classifier.fit(X_train, y_train)

        acc_train = classifier.score(X_train, y_train)
        acc_test = classifier.score(X_test, y_test)
        results_with_best_hyperparams["train_acc"].append(acc_train)
        results_with_best_hyperparams["test_acc"].append(acc_test)

    print("Results with best hyperparams", results_with_best_hyperparams)
    df = pd.DataFrame.from_dict(results_with_best_hyperparams)
    df.to_csv(os.path.join("results/" + path_out))
