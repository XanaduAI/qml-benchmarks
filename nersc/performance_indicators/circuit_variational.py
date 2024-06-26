
from datetime import datetime

import argparse
import csv
import os
import subprocess
import time
import yaml

from pprint import pprint

import jax
import jax.numpy as jnp
import numpy as np

import pennylane as qml

import qml_benchmarks

from qml_benchmarks.hyperparam_search_utils import read_data

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2, 3, 4], help="increase output verbosity",
                        default=1, dest='verb')
    parser.add_argument("--inputPath", default='linearly_separable/', help='input data location')
    parser.add_argument('-n', '--numFeatures', type=int, default=2, help="dataset dimension ")
    parser.add_argument('-m', '--model', help="model: IQPVariationalClassifier, QuantumMetricLearner")

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

    #define model
    if args.model == 'QuantumMetricLearner':
        from qml_benchmarks.models.quantum_metric_learning import QuantumMetricLearner as Model
    elif args.model == 'IQPVariationalClassifier':
        from qml_benchmarks.models.iqp_variational import IQPVariationalClassifier as Model
    else:
        raise ValueError('unknown model %s' % args.model)

    #implementation attributes of model
    use_jax = True
    vmap = True
    jit = True
    max_steps = 100  #the number of gradient descent steps to use to estimate the step time
    model_settings = {'use_jax': use_jax, 'vmap': vmap, 'jit': jit, 'max_steps': max_steps}

    perf_ind_name = 'JAX'  #a name for the performance indicator used for naming files
    n_trials = 1 #number of trials to average over
    n_test = -1 #number of test set points. For full test set use n_test = -1
    #################################

    n_features = args.numFeatures  # dataset dimension
    model_name = Model().__class__.__name__

    # get the 'worst case' hyperparameter settings for the model (those that require the most resources)
    with open('performance_indicators/hyperparam_settings.yaml', "r") as file:
        hp_settings = yaml.safe_load(file)

    hyperparams = {**hp_settings[model_name], **model_settings}
    print(hyperparams)
    
    hyperparams['dev_type'] = 'lightning.qubit'

    assert os.path.exists(args.inputPath)
    # inpF1=f'../../paper/benchmarks/linearly_separable/linearly_separable_{n_features}d_train.csv'
    inpF1 = os.path.join(args.inputPath, 'linearly_separable_%dd_train.csv' % (n_features))
    inpF2 = inpF1.replace('train', 'test')
    print('M:inpF1', inpF1)
    X_train, y_train = read_data(inpF1)
    print('M:inpF2', inpF2)
    X_test, y_test = read_data(inpF2)

    if n_test != -1:
        X_test = X_test[:n_test]
        y_test = y_test[:n_test]

    first_train_steps = []
    av_consec_train_steps = []
    predict_times = []
    
    model = Model(**hyperparams)
    
    def init_circuit(X, y):
        # Derived from: `model.fit(X_train, y_train)`
        model.initialize(n_features=X.shape[1], classes=np.unique(y))
        return model.circuit
    
    circuit = init_circuit(X_train, y_train)
    
    '''
    def initialize_params(self):
        weights = 2 * np.pi * np.random.uniform(size=(self.n_layers, self.n_qubits_, 3))
        weights = jnp.array(weights)
    
        self.params_ = {"weights": weights}
    
    params = model.params_
    x = X_batch
    
    @qml.qnode(dev, **self.qnode_kwargs)
    def circuit(params, x):
        """
        The variational circuit from the plots. Uses an IQP data embedding.
        We use the same observable as in the plots.
        """
        qml.IQPEmbedding(x, wires=range(self.n_qubits_), n_repeats=self.repeats)
        qml.StronglyEntanglingLayers(
            params["weights"], wires=range(self.n_qubits_), imprimitive=qml.CZ
        )
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
    '''

    X = X_train[0]
    y = y_train[0]
    
    def print_elapsed(t1, t2):
        print("%.6f s" % ((t2 - t1).total_seconds()))
    
    specs = qml.specs(circuit, expansion_strategy='device')(model.params_, X)
    #pprint(specs)
    print({
        'num_features': args.numFeatures,
        'device_name': specs['device_name'],
        'gradient_fn': specs['gradient_fn'],
        'num_wires': specs['resources'].num_wires,
        'num_gates': specs['resources'].num_gates,
        'depth': specs['resources'].depth,
        })
    
    print('M:executing circuit')
    t_start = datetime.now()

    for trial in range(n_trials):
        jax.clear_caches()

        expval = circuit(model.params_, X)
        grads = qml.jacobian(circuit)(model.params_, X)

    t_end = datetime.now()
    print_elapsed(t_start, t_end)

    # {'num_features': 15, 'num_wires': 15, 'num_gates': 1800, 'depth': 267}
    # {'num_features': 20, 'num_wires': 20, 'num_gates': 2900, 'depth': 310}
    # {'num_features': 21, 'num_wires': 21, 'num_gates': 3150, 'depth': 321}
    # {'num_features': 22, 'num_wires': 22, 'num_gates': 3410, 'depth': 333}

    # default.qubit
    # 15d - 10 s
    # 20d - 45 s

    # lightning.qubit
    # 15d -  0.4 s
    # 20d -  5 s
    # 21d - 10 s
    # 22d - 22 s

    print('M:done')
