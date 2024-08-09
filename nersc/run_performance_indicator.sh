#!/bin/bash -e

export RAY_DEDUP_LOGS=0

NUM_FEATURES=20

python3 -u performance_indicators/perf_ind_kernel.py --numFeatures $NUM_FEATURES --inputPath performance_indicators/linearly_separable/
