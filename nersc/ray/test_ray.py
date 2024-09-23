"""Example: Run a simple hyperparameter tuning experiment."""

import os
import time
import torch

import ray
from ray import tune, train


def train_function(config: dict) -> None:
    """A training trial function that runs an experiment and saves the results.

    We will run a simple matrix multiplication operation to simulate a
    machine learning model and record some metrics.

    Args:
        config (dict):
            The training configuration (hyperparameters) to run.
    """
    depth = config["depth"]
    width = config["width"]

    # Dummpy initial matrix
    mat = torch.ones((width, width))

    # Runtime for prediction
    start_time = time.time()

    # Dummpy operation to test that things run on GPUs
    for _ in range(depth):
        mat = mat @ mat
    res = torch.trace(mat)
    run_time = time.time() - start_time

    train.report(
        {
            "runtime": run_time,
            "gpus": os.environ["CUDA_VISIBLE_DEVICES"],
            "res": float(res.cpu().numpy()),
        }
    )


if __name__ == "__main__":
    # Initialize and connect to a Ray cluster
    ray.init()

    # Create the hyperparameter search space
    param_space = {
        "depth": tune.grid_search([4, 6, 8]),
        "width": tune.grid_search([2, 4, 8]),
    }

    # Create a Tuner object and start the hyperparameter search
    tuner = tune.Tuner(
        tune.with_resources(train_function, resources={"cpu": 8, "gpu": 1}),
        tune_config=tune.TuneConfig(num_samples=1),
        param_space=param_space,
    )
    tuner.fit()
