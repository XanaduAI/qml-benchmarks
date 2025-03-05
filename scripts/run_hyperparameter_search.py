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

"""Run hyperparameter search and store results with a command-line script."""

import numpy as np
import sys
import os
import time
import argparse
import logging

logging.getLogger().setLevel(logging.INFO)
from importlib import import_module
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from qml_benchmarks.models.base import BaseGenerator
from qml_benchmarks.hyperparam_search_utils import read_data, construct_hyperparameter_grid
from qml_benchmarks.hyperparameter_settings import hyper_parameter_settings

np.random.seed(42)

def custom_scorer(estimator, X, y=None):
    return estimator.score(X, y)

logging.info('cpu count:' + str(os.cpu_count()))

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run experiments with hyperparameter search.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--model",
        help="Model to run",
    )

    parser.add_argument(
        "--dataset-path",
        help="Path to the dataset",
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
        "--hyperparameter-scoring",
        type=list,
        nargs="+",
        default=["accuracy", "roc_auc"],
        help="Scoring for hyperparameter search.",
    )

    parser.add_argument(
        "--hyperparameter-refit",
        type=str,
        default="accuracy",
        help="Refit scoring for hyperparameter search.",
    )

    parser.add_argument(
        "--plot-loss",
        help="True or False. Plot loss history for single fit",
        dest="plot_loss",
        default=False,
        type=bool,
    )

    parser.add_argument(
        "--n-jobs", type=int, default=-1, help="Number of parallel threads to run"
    )

    # Parse the arguments along with any extra arguments that might be model specific
    args, unknown_args = parser.parse_known_args()

    if any(arg is None for arg in [args.model,
                                   args.dataset_path]):
        msg = "\n================================================================================"
        msg += "\nA model from qml.benchmarks.models and dataset path are required. E.g., \n \n"
        msg += "python run_hyperparameter_search \ \n--model DataReuploadingClassifier \ \n--dataset-path train.csv\n"
        msg += "\nCheck all arguments for the script with \n"
        msg += "python run_hyperparameter_search --help\n"
        msg += "================================================================================"
        raise ValueError(msg)

    # Add model specific arguments to override the default hyperparameter grid
    hyperparam_grid = construct_hyperparameter_grid(
        hyper_parameter_settings, args.model
    )

    for hyperparam in hyperparam_grid:
        hp_type = type(hyperparam_grid[hyperparam][0])
        parser.add_argument(f'--{hyperparam}',
                            type=hp_type,
                            nargs="+",
                            default=hyperparam_grid[hyperparam],
                            help=f'{hyperparam} grid values for {args.model}')

    args = parser.parse_args(unknown_args, namespace=args)

    for hyperparam in hyperparam_grid:
        override = getattr(args, hyperparam)
        if override is not None:
            hyperparam_grid[hyperparam] = override
    logging.info(
        "Running hyperparameter search experiment with the following settings\n"
    )
    logging.info(args.model)
    logging.info(args.dataset_path)
    logging.info(" ".join(args.hyperparameter_scoring))
    logging.info(args.hyperparameter_refit)
    logging.info("Hyperparam grid:" + " ".join(
        [(str(key) + str(":") + str(hyperparam_grid[key])) for key in hyperparam_grid.keys()]))

    experiment_path = args.results_path
    results_path = os.path.join(experiment_path, "results")

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    ###################################################################
    # Get the model, dataset and search methods from the arguments
    ###################################################################
    Model = getattr(
        import_module("qml_benchmarks.models"),
        args.model
    )
    model_name = Model.__name__

    is_generative = isinstance(Model(), BaseGenerator)
    use_labels = False if is_generative else True

    # Run the experiments save the results
    train_dataset_filename = os.path.join(args.dataset_path)
    X, y = read_data(train_dataset_filename, labels=use_labels)

    dataset_path_obj = Path(args.dataset_path)
    results_filename_stem = " ".join(
        [Model.__name__ + "_" + dataset_path_obj.stem
         + "_GridSearchCV"])

    # If we have already run this experiment then continue
    if os.path.isfile(os.path.join(results_path, results_filename_stem + ".csv")):
        if args.clean is False:
            msg = "\n================================================================================="
            msg += "\nResults exist in " + os.path.join(results_path, results_filename_stem + ".csv")
            msg += "\nSpecify --clean True to override results or new --results-path"
            msg += "\n================================================================================="
            logging.warning(msg)
            sys.exit(msg)
        else:
            logging.warning("Cleaning existing results for ",
                            os.path.join(results_path, results_filename_stem + ".csv"))

    ###########################################################################
    # Single fit to check everything works
    ###########################################################################
    model = Model()
    a = time.time()
    model.fit(X, y)
    b = time.time()
    default_score = model.score(X, y)
    logging.info(" ".join(
        [model_name,
         "Dataset path",
         args.dataset_path,
         "Train score:",
         str(default_score),
         "Time single run",
         str(b - a)])
    )
    if hasattr(model, "loss_history_"):
        if args.plot_loss:
            plt.plot(model.loss_history_)
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.show()

    if hasattr(model, "n_qubits_"):
        logging.info(" ".join(["Num qubits", f"{model.n_qubits_}"]))

    ###########################################################################
    # Hyperparameter search
    ###########################################################################

    scorer = args.hyperparameter_scoring if not is_generative else custom_scorer
    refit = args.hyperparameter_refit if not is_generative else False

    gs = GridSearchCV(estimator=model, param_grid=hyperparam_grid,
                      scoring=scorer,
                      refit=refit,
                      verbose=3,
                      n_jobs=args.n_jobs).fit(
        X, y
    )
    logging.info("Best hyperparams")
    logging.info(gs.best_params_)

    df = pd.DataFrame.from_dict(gs.cv_results_)
    df.to_csv(os.path.join(results_path, results_filename_stem + ".csv"))

    best_df = pd.DataFrame(list(gs.best_params_.items()), columns=['hyperparameter', 'best_value'])

    # Save best hyperparameters to a CSV file
    best_df.to_csv(os.path.join(results_path,
                                results_filename_stem + '-best-hyperparameters.csv'), index=False)