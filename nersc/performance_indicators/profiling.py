import qml_benchmarks
import pennylane as qml
import jax
import jax.numpy as jnp
import numpy as np
import time
import csv
import os
import pickle
import yaml
import subprocess
from qml_benchmarks.hyperparam_search_utils import read_data

import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2, 3, 4], help="increase output verbosity",
                        default=1, dest='verb')
    parser.add_argument("--inputPath", default='linearly_separable/', help='input data location')
    parser.add_argument('-n', '--numFeatures', type=int, default=2, help="dataset dimension ")

    args = parser.parse_args()

    print('myArg-program:', parser.prog)
    for arg in vars(args):  print('myArg:', arg, getattr(args, arg))

    # assert os.path.exists(args.outPath)
    return args


# =================================
# =================================
#  M A I N
# =================================
# =================================
if __name__ == "__main__":
    args = get_parser()

    ####### SETTINGS #######################
    # You only need to change this to make a different performance indicator

    #define model
    #
    #from qml_benchmarks.models.quantum_metric_learning import QuantumMetricLearner as Model
    from qml_benchmarks.models.iqp_variational import IQPVariationalClassifier as Model
    #implementation attributes of model
    use_jax = True
    vmap = True
    jit = True
    model_settings = {'use_jax': use_jax, 'vmap': vmap, 'jit': jit}

    max_steps = 2 #the number of gradient descent steps to use to estimate the step time
    profile_name = 'jax'  #a name for the performance indicator used for naming files

    #################################

    n_features = args.numFeatures

    model_name = Model().__class__.__name__

    # get the 'worst case' hyperparameter settings for the model (those that require the most resources)
    with open('performance_indicators/hyperparam_settings.yaml', "r") as file:
        hp_settings = yaml.safe_load(file)

    hyperparams = {**hp_settings[model_name], **model_settings}
    print(hyperparams)

    assert os.path.exists(args.inputPath)

    first_step_times = []
    second_step_times = []

    inpF1 = os.path.join(args.inputPath, 'linearly_separable_%dd_train.csv' % (n_features))
    print('M:inpF1', inpF1)
    X_train, y_train = read_data(inpF1)

    jax.clear_caches()
    model = Model(**hyperparams)
    model.fit(X_train, y_train)

    #get step times from loss history data
    step_times = np.array([model.loss_history_[1][i + 1] - model.loss_history_[1][i]
                           for i in range(len(model.loss_history_[1]) - 1)])
    step_times = np.insert(step_times, 0, [model.loss_history_[1][0]])

    file_path = f'performance_indicators/profiling/step_times_{profile_name}_{model_name}.pkl'
    if not os.path.exists(file_path):
        data = {}
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)

    data[n_features] = [step_times[0], step_times[1]]

    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    print('M:done')

