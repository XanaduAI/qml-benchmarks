import pennylane
import qml_benchmarks
from qml_benchmarks.models.iqp_variational import IQPVariationalClassifier
from qml_benchmarks.data.two_curves import generate_two_curves
import numpy as np



X,y = generate_two_curves(100,3, 2, 0.0, 0.1)
model = IQPVariationalClassifier(use_jax=True, vmap=True)

model.fit(X, y)
print(model.loss_history_)
np.savetxt('loss.txt', model.loss_history_)