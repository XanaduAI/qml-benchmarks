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
    #from qml_benchmarks.models.quantum_metric_learning import QuantumMetricLearner as Model
    from qml_benchmarks.models.iqp_variational import IQPVariationalClassifier as Model

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

    for trial in range(n_trials):
        jax.clear_caches()

        model = Model(**hyperparams)

        #note we train for max_steps only, so we won't reach convergence.
        model.fit(X_train, y_train)

        #get step times from loss history data
        step_times = np.array([model.loss_history_[1][i + 1] - model.loss_history_[1][i]
                               for i in range(len(model.loss_history_[1]) - 1)])
        step_times = np.insert(step_times, 0, [model.loss_history_[1][0]])
        print('n_steps:', len(model.loss_history_[1]))

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
    with open('performance_indicators/'+perf_ind_name+'/'+filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in [data]:
            writer.writerow(row)

    # get package list and write to file
    output = subprocess.check_output(['pip', 'list']).decode('utf-8')
    with open('performance_indicators/'+perf_ind_name+'/'+packages_filename, 'w') as file:
        file.write(output)

    print('M:done')

