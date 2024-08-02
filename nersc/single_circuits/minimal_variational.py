
import functools

import numpy as np
from jax import numpy as jnp

import pennylane as qml
import catalyst

from catalyst import qjit

n_qubits_ = 4

x = np.random.rand(n_qubits_)

#shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=n_qubits_)
shape = (2, 4, 3)
params = jnp.array(np.random.random(size=shape))

dev = qml.device("lightning.qubit", wires=n_qubits_)

@qjit
def run_circuit(params, x):

    @qml.qnode(dev)  # diff_method="adjoint" | "finite-diff" | "backprop"
    #@functools.partial(qml.devices.preprocess.decompose, stopping_condition = lambda obj: obj.name not in ['Rot', 'StronglyEntanglingLayers'], max_expansion=3)
    def circuit(params, x):
        qml.IQPEmbedding(x, wires=range(n_qubits_), n_repeats=2)
        qml.StronglyEntanglingLayers(
            params, wires=range(n_qubits_), imprimitive=qml.CZ
        )
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    catalyst.grad(circuit, method="fd")(params, x)  # method="fd"

run_circuit(params, x)
