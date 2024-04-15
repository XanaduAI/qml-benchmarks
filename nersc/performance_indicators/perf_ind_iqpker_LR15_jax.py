import qml_benchmarks
import pennylane as qml
import jax
import jax.numpy as jnp
import numpy as np
import time
import csv
import os
import yaml

from qml_benchmarks.models.iqp_kernel import IQPKernelClassifier
from qml_benchmarks.hyperparam_search_utils import read_data

with open('hyperparam_settings.yaml', "r") as file:
    hp_settings = yaml.safe_load(file)

hyperparams = {**hp_settings['IQPKernelClassifier'], **{'use_jax':True, 'vmap':True}}

print(hyperparams)

n_features = 15 #dataset dimension

X_train,y_train = read_data(f'../../paper/benchmarks/linearly_separable/linearly_separable_{n_features}d_train.csv')
X_test,y_test = read_data(f'../../paper/benchmarks/linearly_separable/linearly_separable_{n_features}d_train.csv')

model = IQPKernelClassifier(**hyperparams)
model.fit(X_train[:50], y_train[:50])

#kernel construction time
construct_kernel_time = model.construct_kernel_time_
#full training time
training_time = model.training_time_
#prediction time
time0 = time.time()
model.predict(X_test[:50])
predict_time = time.time() - time0


#write to csv
data = [construct_kernel_time, training_time, predict_time, hyperparams]

model_name = model.__class__.__name__
filename =  model_name+f"_linearly_separable_{n_features}d_performance_indicators_JAX.csv"
header = ['construct_kernel_time', 'training_time', 'predict_time', 'hyperparameters']

if not os.path.exists('JAX'):
    # Create the directory
    os.mkdir('JAX')

with open('JAX/'+filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    for row in [data]:
        writer.writerow(row)
