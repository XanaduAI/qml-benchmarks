#!/bin/bash -e

REPO=/qml-benchmarks
DATA=${REPO}/nersc/performance_indicators/linearly_separable

GENERATE_DATA=0
# running python paper/benchmarks/generate_linearly_separable.py will generate a folder linearly_separable/ (in the current directory).
if [[ GENERATE_DATA == 1 ]]; then
    python ${REPO}/paper/benchmarks/generate_linearly_separable.py
fi

# You can then use any of the *.csv from this folder to start training. e.g.
#python ${QML}/scripts/run_hyperparameter_search.py\
# --classifier-name IQPVariationalClassifier\
# --dataset-path linearly_separable/linearly_separable_10d_train.csv

# I reduced the grid space and the input size for a faster turn around. This was my command:
python3 ${REPO}/scripts/run_hyperparameter_search.py\
 --classifier-name IQPVariationalClassifier\
 --dataset-path ${DATA}/linearly_separable_4d_train.csv\
 --clean True\
 --n_layers 1 2\
 --learning_rate 0.001\
 --repeats 1\
 --n-jobs 4
