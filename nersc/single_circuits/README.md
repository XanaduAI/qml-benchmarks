
# Benchmarking quantum circuits


## Run with Python `venv`


## Run in `podman` containers 

Prerequisite: Make sure to have datasets available in `single_circuits/linearly_separable`.

Start interactive job on CPU node for testing
``` bash
salloc -q interactive -C cpu -t 0:30:00 -A m4693

# and execute in this interactive session:

IMG=tgermain/ubu22-pennylane-ray

# For preliminary testing whether image is available on node:
CFSH=/global/cfs/cdirs/m4693  # CFS home
REPO_DIR=$CFSH/qml-benchmarks-devel  # qml-benchmark repo
ROOT_DIR=$REPO_DIR/nersc/root  # to access local python packages
WORK_DIR=$REPO_DIR/nersc  # to store output files
# Mount /tmp to avoid following error with Ray:
#     ValueError: Can't find a `node_ip_address.json` file

podman-hpc run -it \
    --net host \
    --volume /tmp:/tmp \
    --volume $ROOT_DIR:/root \
    --volume $REPO_DIR:/qml-benchmarks \
    --volume $WORK_DIR:/work_dir \
    --workdir /work_dir \
    -e HDF5_USE_FILE_LOCKING='FALSE' \
    --shm-size=10.24gb \
    $IMG bash

# Then execute in container, in `work_dir/`:

python3 single_circuits/circuit_variational.py --model IQPVariationalClassifier --numFeatures 21 --inputPath single_circuits/linearly_separable/

python3 single_circuits/demo_variational.py

# exit container

# Run container interactively with wrapper
./wrap_podman.sh $IMG "python3 single_circuits/demo_variational.py"
```
