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

#define model
from qml_benchmarks.models.iqp_variational import IQPVariationalClassifier as Model

#implementation attributes of model
use_jax = True
vmap = True
jit = True

max_steps = 100 #the number of gradient descent steps to use to estimate the step time
perf_ind_name = 'JAX'  #a name for the performance indicator used for naming files
n_features = 2 #dataset dimension
n_trials = 2 #number of trials to average over

#################################

model_name = Model().__class__.__name__

# get the 'worst case' hyperparameter settings for the model (those that require the most resources)
with open('hyperparam_settings.yaml', "r") as file:
    hp_settings = yaml.safe_load(file)

hyperparams = {**hp_settings[model_name], **{'use_jax':use_jax, 'vmap':vmap, 'max_steps':max_steps, 'jit': jit}}
print(hyperparams)

X_train,y_train = read_data(f'../../paper/benchmarks/linearly_separable/linearly_separable_{n_features}d_train.csv')
X_test,y_test = read_data(f'../../paper/benchmarks/linearly_separable/linearly_separable_{n_features}d_train.csv')

first_train_steps = []
av_consec_train_steps = []
predict_times = []

for trial in range(n_trials):
    jax.clear_caches()

    model = Model(**hyperparams)

    #note we train for max_steps only, so we won't reach convergence.
    model.fit(X_train, y_train)

    #get step times from loss history data
    step_times = np.array([model.loss_history_[1][i + 1] - model.loss_history_[1][i]
                           for i in range(len(model.loss_history_[1]) - 1)])
    step_times = np.insert(step_times, 0, [model.loss_history_[1][0]])

    #first train step
    first_train_steps.append(step_times[0])
    #consecutive (average) train step
    av_consec_train_steps.append(float(jnp.mean(step_times[1:])))
    #prediction time
    time0 = time.time()
    model.predict(X_test)
    predict_times.append(time.time() - time0)


#calculate mean and stds
first_train_step = np.mean(first_train_steps)
first_train_step_std = np.std(first_train_steps)

consec_train_step = np.mean(av_consec_train_steps)
consec_train_step_std = np.std(av_consec_train_steps)

predict_time = np.mean(predict_times)
predict_time_std = np.std(predict_times)

#write to csv
data = [first_train_step, first_train_step_std, consec_train_step, consec_train_step_std, predict_time,
        predict_time_std, hyperparams]

filename =  model_name+f"_linearly_separable_{n_features}d_performance_indicators_{perf_ind_name}.csv"
packages_filename =  model_name+f"_linearly_separable_{n_features}d_performance_indicators_{perf_ind_name}_packages.txt"
scontrol_filename =  model_name+f"_linearly_separable_{n_features}d_performance_indicators_{perf_ind_name}_scontrol.txt"

header = ['first_train_step', 'first_train_step_std', 'consec_train_step', 'consec_train_step_std', 'predict_time',
          'predict_time_std', 'hyperparameters']

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

