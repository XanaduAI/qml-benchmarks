import qml_benchmarks
import pennylane as qml
import jax
import jax.numpy as jnp
import numpy as np
import time
import csv
import os
import yaml

from qml_benchmarks.models.quantum_metric_learning import QuantumMetricLearner
from qml_benchmarks.hyperparam_search_utils import read_data

with open('hyperparam_settings.yaml', "r") as file:
    hp_settings = yaml.safe_load(file)

hyperparams = {**hp_settings['QuantumMetricLearner'], **{'use_jax':True, 'vmap':True, 'max_steps':10}}

print(hyperparams)

n_features = 14 #dataset dimension
n_trials = 5 #number of trials to average over

X_train,y_train = read_data(f'../../paper/benchmarks/linearly_separable/linearly_separable_{n_features}d_train.csv')
X_test,y_test = read_data(f'../../paper/benchmarks/linearly_separable/linearly_separable_{n_features}d_train.csv')

first_train_steps = []
av_consec_train_steps = []
predict_times = []

for trial in range(n_trials):
    jax.clear_caches()

    model = QuantumMetricLearner(**hyperparams)
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
    model.predict(X_test[:10])
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

model_name = model.__class__.__name__
filename =  model_name+f"_linearly_separable_{n_features}d_performance_indicators_JAX.csv"
header = ['first_train_step', 'first_train_step_std', 'consec_train_step', 'consec_train_step_std', 'predict_time',
          'predict_time_std', 'hyperparameters']

if not os.path.exists('JAX'):
    # Create the directory
    os.mkdir('JAX')

with open('JAX/'+filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    for row in [data]:
        writer.writerow(row)
