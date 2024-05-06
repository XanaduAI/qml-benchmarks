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
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3,4],  help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("--inputPath",default='linearly_separable/',help='input data location')
    parser.add_argument('-n','--numFeatures',type=int,default=2, help="dataset dimension ")

    args = parser.parse_args()
    
    print( 'myArg-program:',parser.prog)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))

    #assert os.path.exists(args.outPath)
    return args

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    args=get_parser()

    #################################
    # settings for the performance indicator.
    # You only need to change this to make a different performance indicator

    #define the model
    from qml_benchmarks.models.iqp_kernel import IQPKernelClassifier as Model

    #implementation attributes of model
    use_jax = True
    vmap = True
    jit = True

    perf_ind_name = 'JAX'  #a name for the performance indicator used for naming files
    n_features = args.numFeatures  #dataset dimension

    #################################

    model_name = Model().__class__.__name__

    # get the 'worst case' hyperparameter settings for the model (those that require the most resources)
    with open('performance_indicators/hyperparam_settings.yaml', "r") as file:
        hp_settings = yaml.safe_load(file)

    hyperparams = {**hp_settings[model_name], **{'use_jax':use_jax, 'vmap':vmap, 'jit': jit}}
    print(hyperparams)
    assert os.path.exists(args.inputPath)
    #inpF1=f'../../paper/benchmarks/linearly_separable/linearly_separable_{n_features}d_train.csv'
    inpF1=os.path.join(args.inputPath,'linearly_separable_%dd_train.csv'%(n_features))
    inpF2=inpF1.replace('train','test')
    print('M:inpF1',inpF1)
    X_train,y_train = read_data(inpF1)
    print('M:inpF2',inpF2)
    X_test,y_test = read_data(inpF2)

    model = Model(**hyperparams)
    model.fit(X_train, y_train)

    #kernel construction time
    construct_kernel_time = model.construct_kernel_time_
    #full training time
    training_time = model.training_time_
    #prediction time
    time0 = time.time()
    model.predict(X_test)
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
