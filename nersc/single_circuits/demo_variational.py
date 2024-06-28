
'''
https://docs.ray.io/en/latest/ray-core/tasks.html#ray-remote-functions
https://docs.ray.io/en/latest/ray-core/patterns/limit-running-tasks.html
'''

import numpy as np
import time

import pennylane as qml

# TODO: fix hanging run with qml.np <GRAD>
#from pennylane import numpy as np

from datetime import datetime

def print_elapsed(t1, t2):
    print("%.6f s" % ((t2 - t1).total_seconds()))

# Model parameters available in config originate from circuit_variational.py.

config = {
    'device': 'lightning.qubit',
    #'n_features': 15, 'n_layers': 15, 'n_repeats': 10, 'n_params': 1, 'sample_shape': (15,), 'num_wires': 15, 'num_gates': 1800, 'depth': 267
    'n_features': 20, 'n_layers': 15, 'n_repeats': 10, 'n_params': 1, 'sample_shape': (20,), 'num_wires': 20, 'num_gates': 2900, 'depth': 310
    #'n_features': 21, 'n_layers': 15, 'n_repeats': 10, 'n_params': 1, 'sample_shape': (21,), 'num_wires': 21, 'num_gates': 3150, 'depth': 321
    #'n_features': 22, 'n_layers': 15, 'n_repeats': 10, 'n_params': 1, 'sample_shape': (22,), 'num_wires': 22, 'num_gates': 3410, 'depth': 333
    #'n_features': 23, 'n_layers': 15, 'n_repeats': 10, 'n_params': 1, 'sample_shape': (23,), 'num_wires': 23, 'num_gates': 3680, 'depth': 346
    #'n_features': 24, 'n_layers': 15, 'n_repeats': 10, 'n_params': 1, 'sample_shape': (24,), 'num_wires': 24, 'num_gates': 3960, 'depth': 358
    #'n_features': 25, 'n_layers': 15, 'n_repeats': 10, 'n_params': 1, 'sample_shape': (25,), 'num_wires': 25, 'num_gates': 4250, 'depth': 371
}

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
        weights = np.array(weights)  # requires_grad=True  # <GRAD>

        self.params_ = {"weights": weights}

    def create_circuit(self, x):

        @qml.qnode(dev)
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

specs = qml.specs(circuit, expansion_strategy='device')(model.params_, X)
print({
    'n_features': n_features,
    'n_layers': model.n_layers,
    'n_repeats': model.repeats,
    'n_params': len(model.params_),
    'sample_shape': X.shape,
    'device_name': specs['device_name'],
    'gradient_fn': specs['gradient_fn'],
    'num_wires': specs['resources'].num_wires,
    'num_gates': specs['resources'].num_gates,
    'depth': specs['resources'].depth,
    })

print('running circuit()')
t_start = datetime.now()
expval = circuit(model.params_, X)
# TODO: activate gradients <GRAD>
#grads = qml.jacobian(circuit)(model.params_, X)
t_end = datetime.now()

print(expval)

print_elapsed(t_start, t_end)
