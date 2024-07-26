
'''
https://docs.ray.io/en/latest/ray-core/tasks.html#ray-remote-functions
https://docs.ray.io/en/latest/ray-core/patterns/limit-running-tasks.html
'''

import argparse
import time

import numpy as np

import pennylane as qml

from pennylane import numpy as np  # <GRAD>

from datetime import datetime

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--numFeatures', type=int, default=15, help="dataset dimension ")
    parser.add_argument('-q', '--device', default='lightning.qubit', help="quantum device e.g. lightning.qubit")
    parser.add_argument('-g', '--gradients', action='store_true', help="request gradients wrt. all weights")
    parser.add_argument('-d', '--dryRun', action='store_true', help="print specs only, no circuit execution")
    args = parser.parse_args()
    return args

args = get_parser()

def print_elapsed(t1, t2):
    print("%.6f s" % ((t2 - t1).total_seconds()))

# Model parameters available in config originate from circuit_variational.py.
catalog = {
    10: {'n_features': 10, 'n_layers': 15, 'n_repeats': 10, 'n_params': 1, 'sample_shape': (10,), 'num_wires': 10, 'num_gates': 950, 'depth': 198},
    15: {'n_features': 15, 'n_layers': 15, 'n_repeats': 10, 'n_params': 1, 'sample_shape': (15,), 'num_wires': 15, 'num_gates': 1800, 'depth': 267},
    16: {'n_features': 16, 'n_layers': 15, 'n_repeats': 10, 'n_params': 1, 'sample_shape': (16,), 'num_wires': 16, 'num_gates': 2000, 'depth': 281},
    17: {'n_features': 17, 'n_layers': 15, 'n_repeats': 10, 'n_params': 1, 'sample_shape': (17,), 'num_wires': 17, 'num_gates': 2210, 'depth': 282},
    18: {'n_features': 18, 'n_layers': 15, 'n_repeats': 10, 'n_params': 1, 'sample_shape': (18,), 'num_wires': 18, 'num_gates': 2430, 'depth': 289},
    19: {'n_features': 19, 'n_layers': 15, 'n_repeats': 10, 'n_params': 1, 'sample_shape': (19,), 'num_wires': 19, 'num_gates': 2660, 'depth': 299},
    20: {'n_features': 20, 'n_layers': 15, 'n_repeats': 10, 'n_params': 1, 'sample_shape': (20,), 'num_wires': 20, 'num_gates': 2900, 'depth': 310},
    21: {'n_features': 21, 'n_layers': 15, 'n_repeats': 10, 'n_params': 1, 'sample_shape': (21,), 'num_wires': 21, 'num_gates': 3150, 'depth': 321},
    22: {'n_features': 22, 'n_layers': 15, 'n_repeats': 10, 'n_params': 1, 'sample_shape': (22,), 'num_wires': 22, 'num_gates': 3410, 'depth': 333},
    23: {'n_features': 23, 'n_layers': 15, 'n_repeats': 10, 'n_params': 1, 'sample_shape': (23,), 'num_wires': 23, 'num_gates': 3680, 'depth': 346},
    24: {'n_features': 24, 'n_layers': 15, 'n_repeats': 10, 'n_params': 1, 'sample_shape': (24,), 'num_wires': 24, 'num_gates': 3960, 'depth': 358},
    25: {'n_features': 25, 'n_layers': 15, 'n_repeats': 10, 'n_params': 1, 'sample_shape': (25,), 'num_wires': 25, 'num_gates': 4250, 'depth': 371},
    26: {'n_features': 26, 'n_layers': 15, 'n_repeats': 10, 'n_params': 1, 'sample_shape': (26,), 'num_wires': 26, 'num_gates': 4550, 'depth': 383},
    27: {'n_features': 27, 'n_layers': 15, 'n_repeats': 10, 'n_params': 1, 'sample_shape': (27,), 'num_wires': 27, 'num_gates': 4860, 'depth': 396},
    28: {'n_features': 28, 'n_layers': 15, 'n_repeats': 10, 'n_params': 1, 'sample_shape': (28,), 'num_wires': 28, 'num_gates': 5180, 'depth': 409},
    29: {'n_features': 29, 'n_layers': 15, 'n_repeats': 10, 'n_params': 1, 'sample_shape': (29,), 'num_wires': 29, 'num_gates': 5510, 'depth': 423},
    30: {'n_features': 30, 'n_layers': 15, 'n_repeats': 10, 'n_params': 1, 'sample_shape': (30,), 'num_wires': 30, 'num_gates': 5850, 'depth': 435},
}

config = dict(catalog[args.numFeatures])

config['device'] = args.device

n_features = config['n_features']
n_layers = config['n_layers']
n_repeats = config['n_repeats']

dev = qml.device(config['device'], wires=n_features)

class VariationalModel:

    def __init__(self, n_features, n_layers, n_repeats, x):
        self.n_qubits_ = n_features
        self.n_layers = n_layers
        self.repeats = n_repeats
        self.params_ = None

        self.initialize_params()
        self.create_circuit(x)

    def initialize_params(self):
        weights = 2 * np.pi * np.random.uniform(size=(self.n_layers, self.n_qubits_, 3))
        weights = np.array(weights, requires_grad=True)  # <GRAD>

        self.params_ = {"weights": weights}

    def create_circuit(self, x):

        @qml.qnode(dev)  # diff_method="adjoint"  # <GRAD>
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
        
        self.circuit = circuit

X = np.random.rand(*config['sample_shape'])

model = VariationalModel(n_features, n_layers, n_repeats, X)
circuit = model.circuit

if args.dryRun:
    print('inspecting circuit()')
    specs = qml.specs(circuit, expansion_strategy='device')(model.params_, X)
    print({
        'n_features': n_features,
        'n_layers': model.n_layers,
        'n_repeats': model.repeats,
        'n_params': model.params_["weights"].size,
        'sample_shape': X.shape,
        'device_name': specs['device_name'],
        'gradient_fn': specs['gradient_fn'],
        'num_wires': specs['resources'].num_wires,
        'num_gates': specs['resources'].num_gates,
        'depth': specs['resources'].depth,
        })
    exit(0)

print('running circuit()')
t_start = datetime.now()
expval = circuit(model.params_, X)
if args.gradients:
    grads = qml.jacobian(circuit)(model.params_, X)
t_end = datetime.now()

#print(expval)

print_elapsed(t_start, t_end)
