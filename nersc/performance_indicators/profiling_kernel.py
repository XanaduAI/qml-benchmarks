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
    from qml_benchmarks.models.iqp_kernel import IQPKernelClassifier as Model
    #implementation attributes of model
    use_jax = False
    vmap = False
    jit = True
    model_settings = {'use_jax': use_jax, 'vmap': vmap, 'jit': jit}

    profile_name = 'catalyst_qjit'  #a name for the performance indicator used for naming files

    #################################

    n_features = args.numFeatures

    print('NUM FEATURES: ' + str(n_features))

    model_name = Model().__class__.__name__

    # get the 'worst case' hyperparameter settings for the model (those that require the most resources)
    with open('performance_indicators/hyperparam_settings.yaml', "r") as file:
        hp_settings = yaml.safe_load(file)

    hyperparams = {**hp_settings[model_name], **model_settings}
    print(hyperparams)

    assert os.path.exists(args.inputPath)

    av_circuit_times = [] #the average time after the first circuit has been compiled

    inpF1 = os.path.join(args.inputPath, 'linearly_separable_%dd_train.csv' % (n_features))
    print('M:inpF1', inpF1)
    X_train, y_train = read_data(inpF1)

    jax.clear_caches()
    model = Model(**hyperparams)

    if model_name=='ProjectedQuantumKernel':
        first_circuit_time = model.circuit(X[0])
        second_circuit_time = model.circuit(X[0])
    elif model_name == 'IQPKernelClassifier':
        first_circuit_time = model.circuit(jnp.concatenate((X[0], X[1])))
        second_circuit_time = model.circuit(jnp.concatenate((X[0], X[1])))

    dir_path = f'performance_indicators/profiling'
    file_path = f'{dir_path}/step_times_{profile_name}_{model_name}.pkl'

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    if not os.path.exists(file_path):
        data = {}
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)

    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    data[n_features] = [first_circuit_time, second_circuit_time]

    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

    print('M:done')

