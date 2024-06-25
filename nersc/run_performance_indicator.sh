#!/bin/bash -e

export RAY_DEDUP_LOGS=0

numFeatures=4

python3 -u performance_indicators/perf_ind_kernel.py --numFeatures $numFeatures --inputPath performance_indicators/linearly_separable/
