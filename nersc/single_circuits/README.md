
# Benchmarking quantum circuits


## Run with Python `venv`

### `lightning-kokkos` from pypi wheels

Python venv with pypi wheels
```
cd /global/common/software/m4693/

module load python
mkdir -p venv
python -m venv venv/qml_LK
source venv/qml_LK/bin/activate

pip install pennylane-lightning
pip install pennylane-lightning[kokkos]

pip install pennylane-catalyst

cd /global/cfs/cdirs/m4693/qml-benchmarks-devel
pip install --user .
```

Start interactive job on CPU node for testing
``` bash
salloc -q interactive -C cpu -t 0:30:00 -A m4693

# and execute in this interactive session:

source /global/common/software/m4693/venv/qml_LK/bin/activate
cd nersc/

# to restrict the number of threads:
#export OMP_NUM_THREADS=32

python3 single_circuits/demo_variational.py -n 25
```

Stats on interactive CPU node (nid004079)
```
lightning.qubit
  15d -      s
  20d -  3.3 s
  21d -  7 s
  22d - 16 s
  23d - 35 s
lightning.kokkos
  23d -  1 s
  25d -  5 s (7 s with 32 threads)
  26d - 34 s
```

### `lightning-kokkos` from source with CUDA

lightning-kokkos with GPU
- https://pypi.org/project/PennyLane-Lightning-Kokkos/
- https://docs.pennylane.ai/projects/lightning/en/stable/lightning_kokkos/installation.html
- https://github.com/PennyLaneAI/lightning-on-hpc/blob/main/DataCollection/distributed/LUMI_LKOKKOS_VQE/README.md- 

```
cd /global/common/software/m4693/

module load cudatoolkit

module load python
mkdir -p venv
python -m venv venv/qml_LK_GPU
source venv/qml_LK_GPU/bin/activate

python -m pip install pip==22.0

git clone https://github.com/PennyLaneAI/pennylane-lightning.git
cd pennylane-lightning

git checkout v0.36.0

pip install -r requirements.txt

# install lightning-qubit as prerequisite
CXX=$(which CC) python -m pip install -e . --verbose

CXX=$(which CC) CMAKE_ARGS="-DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80:BOOL=ON -DCMAKE_CXX_COMPILER=$(which CC)" PL_BACKEND="lightning_kokkos" python -m pip install . --verbose
```

Start interactive job on CPU node for testing
``` bash
salloc -q interactive -C gpu -t 0:30:00 -A m4693

# and execute in this interactive session:

source /global/common/software/m4693/venv/qml_LK_GPU/bin/activate
cd nersc/

# to restrict the number of threads:
export OMP_NUM_THREADS=1

python3 single_circuits/demo_variational.py -n 25
```

Stats on interactive GPU node (nid200381)
```
lightning.kokkos
  23d -    s
  25d -  3 s
  26d -  6 s
  27d - 12 s
  28d - 25 s
```

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
