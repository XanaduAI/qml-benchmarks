import qml_benchmarks
import pennylane as qml
import jax
import jax.numpy as jnp
import numpy as np
import time
import csv
import os
import yaml
import subprocess
from qml_benchmarks.hyperparam_search_utils import read_data

#################################
# settings for the performance indicator.
# You only need to change this to make a different performance indicator

from qml_benchmarks.models.iqp_kernel import IQPKernelClassifier as Model
use_jax = True
vmap = True
jit = True
perf_ind_name = 'JAX'  #a name for the performance indicator used for naming files
n_features = 3 #dataset dimension
n_train = 50 #number of data points to fit model
n_test = 50 #number of datapoints for prediction

#################################

model_name = Model().__class__.__name__

# get the 'worst case' hyperparameter settings for the model (those that require the most resources)
with open('hyperparam_settings.yaml', "r") as file:
    hp_settings = yaml.safe_load(file)

hyperparams = {**hp_settings[model_name], **{'use_jax':use_jax, 'vmap':vmap, 'jit': jit}}
print(hyperparams)

X_train,y_train = read_data(f'../../paper/benchmarks/linearly_separable/linearly_separable_{n_features}d_train.csv')
X_test,y_test = read_data(f'../../paper/benchmarks/linearly_separable/linearly_separable_{n_features}d_train.csv')

model = Model(**hyperparams)
model.fit(X_train[:n_train], y_train[:n_train])

#kernel construction time
construct_kernel_time = model.construct_kernel_time_
#full training time
training_time = model.training_time_
#prediction time
time0 = time.time()
model.predict(X_test[:n_test])
predict_time = time.time() - time0


#write to csv
data = [construct_kernel_time, training_time, predict_time, hyperparams]

filename =  model_name+f"_linearly_separable_{n_features}d_performance_indicators_{perf_ind_name}.csv"
packages_filename =  model_name+f"_linearly_separable_{n_features}d_performance_indicators_{perf_ind_name}_packages.txt"
scontrol_filename =  model_name+f"_linearly_separable_{n_features}d_performance_indicators_{perf_ind_name}_scontrol.txt"

header = ['construct_kernel_time', 'training_time', 'predict_time', 'hyperparameters']

if not os.path.exists(perf_ind_name):
    # Create the directory
    os.mkdir(perf_ind_name)

#write perf indicator data
with open(perf_ind_name+'/'+filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    for row in [data]:
        writer.writerow(row)

# get package list and write to file
output = subprocess.check_output(['pip', 'list']).decode('utf-8')
with open(perf_ind_name+'/'+packages_filename, 'w') as file:
    file.write(output)

# make an empty text file to store the scontrol data
with open(perf_ind_name+'/'+scontrol_filename, 'w') as file:
    pass