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
from importlib import import_module
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
# Import BaseGenerator later only if needed, reducing top-level dependencies
# from qml_benchmarks.models.base import BaseGenerator
from qml_benchmarks.hyperparam_search_utils import read_data, construct_hyperparameter_grid
from qml_benchmarks.hyperparameter_settings import hyper_parameter_settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(asctime)s:%(message)s')

np.random.seed(42)

# --- Framework Availability Check (Informational) ---
print("-" * 50)
logging.info("Checking QML Framework Availability...")

PENN LANE_AVAILABLE = False
try:
    import pennylane as qml
    PENN LANE_AVAILABLE = True
    logging.info(f"  ✅ PennyLane v{getattr(qml, '__version__', 'N/A')} found.")
except ImportError:
    logging.warning("  ⚠️ PennyLane not found or failed to import.")
except RuntimeError as e: # Catch potential init errors
    logging.warning(f"  ⚠️ PennyLane import failed with RuntimeError: {e}.")
except Exception as e:
    logging.warning(f"  ⚠️ Unexpected error importing PennyLane: {type(e).__name__}: {e}.")


TFQ_AVAILABLE = False
TF_AVAILABLE = False
try:
    import tensorflow as tf
    TF_AVAILABLE = True
    logging.info(f"  ✅ TensorFlow v{getattr(tf, '__version__', 'N/A')} found.")
    # Check TFQ only if TF is available
    try:
        import tensorflow_quantum as tfq
        TFQ_AVAILABLE = True
        logging.info(f"  ✅ TensorFlow Quantum v{getattr(tfq, '__version__', 'N/A')} found.")
    except ImportError:
        logging.warning("  ⚠️ TensorFlow Quantum not found or failed to import.")
    except Exception as e:
         logging.warning(f"  ⚠️ TensorFlow Quantum import failed with error: {type(e).__name__}: {e}.")
except ImportError:
    logging.warning("  ⚠️ TensorFlow not found or failed to import (TFQ requires TF).")
except Exception as e: # Catch other potential TF errors like the CUDA ones
     logging.warning(f"  ⚠️ TensorFlow import failed with error: {type(e).__name__}: {e}.")

# Add TorchQuantum check if relevant for any models used by this script
TORCHQUANTUM_AVAILABLE = False
try:
    import torch # Check if torch exists first
    # Import TorchQuantum (adjust import name if different)
    import torchquantum as tq
    TORCHQUANTUM_AVAILABLE = True
    logging.info(f"  ✅ TorchQuantum v{getattr(tq, '__version__', 'N/A')} found (needs PyTorch v{getattr(torch, '__version__', 'N/A')}).")
except ImportError:
     logging.warning("  ⚠️ TorchQuantum or PyTorch not found/failed to import.")
except RuntimeError as e: # Catch potential init errors like Triton
    logging.warning(f"  ⚠️ TorchQuantum/PyTorch import failed with RuntimeError: {e}.")
except Exception as e:
    logging.warning(f"  ⚠️ Unexpected error importing TorchQuantum/PyTorch: {type(e).__name__}: {e}.")

print("-" * 50)
# --- End Framework Availability Check ---


def custom_scorer(estimator, X, y=None):
    # Make sure scorer works even if BaseGenerator wasn't imported earlier
    try:
        from qml_benchmarks.models.base import BaseGenerator # Import only if needed
        if isinstance(estimator, BaseGenerator):
            # Generative models might have a different scoring logic
            # Placeholder: return a dummy score or implement appropriate generative metric
            return 0.0 # Needs proper implementation if generative models are used
        else:
             return estimator.score(X, y)
    except ImportError:
         # Fallback if BaseGenerator cannot be imported
         logging.warning("Could not import BaseGenerator for custom scorer check.")
         return estimator.score(X, y)


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run experiments with hyperparameter search.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # --- Argument Parsing (remains the same) ---
    parser.add_argument(
        "--model",
        help="Model class name to run (must exist in src/qml_benchmarks/models/)",
        required=True # Make model required
    )
    parser.add_argument(
        "--dataset-path",
        help="Path to the training dataset (e.g., train.csv)",
        required=True # Make dataset path required
    )
    parser.add_argument(
        "--results-path", default=".", help="Path to store the experiment results"
    )
    parser.add_argument(
        "--clean",
        help="True or False. Remove previous results if they exist.",
        action=argparse.BooleanOptionalAction, # Use boolean action
        default=False
    )
    parser.add_argument(
        "--hyperparameter-scoring",
        type=str, # Read as string, parse later if needed
        nargs="+",
        default=["accuracy"], # Default to accuracy for simplicity unless known otherwise
        help="Scoring metric(s) for hyperparameter search (e.g., accuracy roc_auc).",
    )
    parser.add_argument(
        "--hyperparameter-refit",
        type=str,
        default="accuracy", # Should match one of the scoring metrics
        help="Refit scoring metric for hyperparameter search.",
    )
    parser.add_argument(
        "--plot-loss",
        help="Plot loss history after single fit (if available).",
        action=argparse.BooleanOptionalAction,
        default=False
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1, help="Number of parallel jobs for GridSearchCV (-1 uses all processors)."
    )

    # Parse known arguments first
    args, unknown_args = parser.parse_known_args()

    # --- Load Model Class with Error Handling ---
    model_name = args.model
    ModelClass = None
    logging.info(f"Attempting to load model class: {model_name}...")
    try:
        # Dynamically import the requested model class
        model_module = import_module("qml_benchmarks.models")
        if not hasattr(model_module, model_name):
             raise AttributeError(f"Model class '{model_name}' not found in qml_benchmarks.models module.")
        ModelClass = getattr(model_module, model_name)
        logging.info(f"Successfully located model class: {model_name}")

        # Quick instantiation check to catch init-time errors if possible
        # This might still fail later if fit() uses specific frameworks
        _ = ModelClass() # Try creating an instance
        logging.info("Basic model instantiation check passed.")

    except ImportError as e:
        # Check if the error is due to a known missing framework
        missing_module_str = str(e).split("'")[-2] if "'" in str(e) else "Unknown"
        error_msg = f"Failed to import model '{model_name}' or its dependencies: {e}\n"
        framework_needed = "Unknown"

        if "pennylane" in missing_module_str.lower():
             framework_needed = "PennyLane"
        elif "tensorflow_quantum" in missing_module_str.lower() or ("tensorflow" in missing_module_str.lower() and "quantum" in args.model.lower()):
             framework_needed = "TensorFlow/TensorFlow Quantum"
        elif "torchquantum" in missing_module_str.lower(): # Add if needed
             framework_needed = "TorchQuantum/PyTorch"
        elif "torch" in missing_module_str.lower(): # General PyTorch if needed by PL model
             framework_needed = "PyTorch"

        if framework_needed != "Unknown":
             error_msg += f"Could not run model '{model_name}'. The required framework '{framework_needed}' is not installed or failed to import correctly.\n"
             error_msg += f"Please install '{framework_needed}' following its official documentation and ensure its dependencies (like TF or PyTorch) are compatible."
        else:
             error_msg += f"Could not run model '{model_name}'. Failed to import dependency '{missing_module_str}'. Please ensure all core requirements are installed."

        logging.error(error_msg)
        sys.exit(1) # Exit if the requested model cannot be loaded

    except AttributeError as e:
        logging.error(f"Model class '{model_name}' not found in the 'qml_benchmarks.models' module. Check spelling.")
        logging.error(f"Details: {e}")
        sys.exit(1)
    except Exception as e:
         logging.error(f"Failed to load or instantiate model class '{model_name}': {type(e).__name__}: {e}")
         # import traceback # Uncomment for detailed debugging
         # traceback.print_exc()
         sys.exit(1) # Exit on other loading errors

    # --- Construct Hyperparameter Grid ---
    # (Remains mostly the same, assuming ModelClass is now loaded)
    logging.info("Constructing hyperparameter grid...")
    hyperparam_grid = construct_hyperparameter_grid(
        hyper_parameter_settings, model_name # Use model_name string here
    )

    # Override grid with any command-line arguments provided
    # Re-parse arguments to include model-specific ones added dynamically IF NEEDED
    # However, it's usually better to define model-specific args separately if truly needed
    # For now, let's assume the initial grid is sufficient or models handle extra args internally
    # (Re-adding dynamic parser args here can be complex and error-prone)

    logging.info(f"Using Hyperparameter Grid for {model_name}:")
    for key, values in hyperparam_grid.items():
         logging.info(f"  {key}: {values}")

    # --- Prepare Paths and Data ---
    # (Remains the same)
    logging.info("Preparing results paths and reading data...")
    experiment_path = args.results_path
    results_path = os.path.join(experiment_path, "results")

    if not os.path.exists(results_path):
        try:
            os.makedirs(results_path)
            logging.info(f"Created results directory: {results_path}")
        except OSError as e:
            logging.error(f"Could not create results directory {results_path}: {e}")
            sys.exit(1)

    # Check if model requires labels (classification vs generation)
    # Instantiate the model *once* to check its type
    try:
        temp_model_instance = ModelClass()
        # Check if it inherits from BaseGenerator correctly
        try:
             from qml_benchmarks.models.base import BaseGenerator
             is_generative = isinstance(temp_model_instance, BaseGenerator)
        except ImportError:
             logging.warning("Could not import BaseGenerator, assuming model is not generative.")
             is_generative = False
        use_labels = not is_generative
        logging.info(f"Model '{model_name}' requires labels: {use_labels}")
    except Exception as e:
         logging.error(f"Failed to instantiate model '{model_name}' to check type: {e}")
         sys.exit(1)


    # Read data
    try:
        train_dataset_filename = os.path.join(args.dataset_path)
        logging.info(f"Reading data from: {train_dataset_filename}")
        X, y = read_data(train_dataset_filename, labels=use_labels)
        logging.info(f"Data loaded successfully. X shape: {X.shape}, y shape: {y.shape if y is not None else 'N/A'}")
    except FileNotFoundError:
        logging.error(f"Dataset file not found at: {train_dataset_filename}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to read data from {train_dataset_filename}: {e}")
        sys.exit(1)


    # --- Check for Existing Results ---
    # (Remains the same)
    dataset_path_obj = Path(args.dataset_path)
    results_filename_stem = f"{model_name}_{dataset_path_obj.stem}_GridSearchCV" # Use f-string
    results_csv_path = os.path.join(results_path, results_filename_stem + ".csv")
    best_params_csv_path = os.path.join(results_path, results_filename_stem + "-best-hyperparameters.csv")

    if os.path.isfile(results_csv_path):
        if not args.clean: # Check the boolean action result
            msg = f"\nResults already exist in {results_csv_path}\n"
            msg += "Specify --clean to override results or use a different --results-path\n"
            logging.warning(msg)
            sys.exit(0) # Exit successfully as results exist
        else:
            logging.warning(f"Cleaning existing results file: {results_csv_path}")
            try:
                os.remove(results_csv_path)
                if os.path.exists(best_params_csv_path):
                     os.remove(best_params_csv_path)
            except OSError as e:
                 logging.error(f"Could not remove existing results files: {e}")
                 # Continue execution, GridSearchCV might handle overwriting or error out

    # --- Single Fit Check ---
    # (Remains mostly the same, but uses ModelClass)
    logging.info("\nPerforming single fit check...")
    model_instance_single = ModelClass() # Use the loaded class
    start_time = time.time()
    try:
        model_instance_single.fit(X, y)
        end_time = time.time()
        default_score = model_instance_single.score(X, y) # Use instance scoring
        logging.info(
            f"{model_name} | Dataset: {args.dataset_path} | "
            f"Single Run Train Score: {default_score:.4f} | Time: {end_time - start_time:.2f}s"
        )

        if hasattr(model_instance_single, "loss_history_") and args.plot_loss:
            try:
                plt.figure() # Create a new figure
                plt.plot(model_instance_single.loss_history_)
                plt.xlabel("Iterations")
                plt.ylabel("Loss")
                plt.title(f"Loss History for {model_name} (Single Fit)")
                plot_path = os.path.join(results_path, f"{results_filename_stem}_single_fit_loss.png")
                plt.savefig(plot_path)
                plt.close() # Close plot to avoid displaying in non-interactive environments
                logging.info(f"Saved single fit loss plot to {plot_path}")
            except Exception as plot_e:
                logging.warning(f"Could not plot/save loss history: {plot_e}")

        if hasattr(model_instance_single, "n_qubits_"):
            logging.info(f"Num qubits used by model: {model_instance_single.n_qubits_}")

    except Exception as fit_e:
        logging.error(f"Error during single fit check for model '{model_name}': {fit_e}")
        logging.error("This might indicate issues with the model implementation or data compatibility.")
        # import traceback # Uncomment for detailed debugging
        # traceback.print_exc()
        sys.exit(1) # Exit if the single fit fails

    # --- Hyperparameter Search ---
    # (Remains mostly the same, uses ModelClass)
    logging.info("\nStarting Hyperparameter Search (GridSearchCV)...")

    # Re-instantiate model for GridSearchCV
    model_for_grid = ModelClass()

    # Adjust scorer and refit based on generative flag
    # Note: custom_scorer might need actual implementation if generative models are used
    scorer = args.hyperparameter_scoring if not is_generative else make_scorer(custom_scorer)
    refit_strategy = args.hyperparameter_refit if not is_generative else False
    # Ensure refit is one of the scorers if not False
    if isinstance(scorer, list) and refit_strategy not in scorer and refit_strategy is not False:
        logging.warning(f"Refit metric '{refit_strategy}' not in scorers {scorer}. Setting refit=False.")
        refit_strategy = False
    elif isinstance(scorer, str) and refit_strategy != scorer and refit_strategy is not False:
         logging.warning(f"Refit metric '{refit_strategy}' does not match scorer '{scorer}'. Setting refit=False.")
         refit_strategy = False


    try:
        gs = GridSearchCV(
            estimator=model_for_grid,
            param_grid=hyperparam_grid,
            scoring=scorer,
            refit=refit_strategy,
            verbose=3,
            n_jobs=args.n_jobs,
            error_score='raise' # Raise errors during grid search fits
        ).fit(X, y) # Pass data

        logging.info("\nGridSearchCV Finished.")
        logging.info(f"Best hyperparams found for refit='{refit_strategy}':")
        logging.info(gs.best_params_)
        if refit_strategy:
             logging.info(f"Best score ({refit_strategy}): {gs.best_score_:.4f}")


        df = pd.DataFrame.from_dict(gs.cv_results_)
        df.to_csv(results_csv_path, index=False)
        logging.info(f"Full GridSearchCV results saved to: {results_csv_path}")

        best_df = pd.DataFrame(list(gs.best_params_.items()), columns=['hyperparameter', 'best_value'])
        best_df.to_csv(best_params_csv_path, index=False)
        logging.info(f"Best hyperparameters saved to: {best_params_csv_path}")

    except Exception as gs_e:
         logging.error(f"Error during GridSearchCV for model '{model_name}': {gs_e}")
         logging.error("Check the traceback for issues within model fitting or scoring during the search.")
         # import traceback # Uncomment for detailed debugging
         # traceback.print_exc()
         sys.exit(1) # Exit if grid search fails

    logging.info("Experiment finished successfully.")