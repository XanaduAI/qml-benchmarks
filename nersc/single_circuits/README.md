
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

cd /global/cfs/cdirs/m4693/qml-benchmarks-devel
pip install -e .  # --user

pip install ray  # for other experiments

pip install pennylane-lightning
pip install pennylane-lightning[kokkos]

pip install pennylane-catalyst
```

Start interactive job on CPU node for testing
``` bash
salloc -q interactive -C cpu -t 0:30:00 -A m4693

# and execute in this interactive session:

source /global/common/software/m4693/venv/qml_LK/bin/activate
cd nersc/

# to restrict the number of threads:
export OMP_NUM_THREADS=32
python3 single_circuits/demo_variational.py -q lightning.qubit -n 25
```

Stats on interactive CPU node (nid004079)
```
> Weights as native numpy arrays
lightning.qubit
  15 -  0.1 s
  20 -  3.3 s
  21 -  7 s
  22 - 16 s
  23 - 35 s
lightning.kokkos
  23 -  1 s
  25 -  5 s (7 s with 32 threads)
  26 - 34 s

> Benchmarking numpy/qml.numpy, gradients
> no-grad: qml.np.array(requires_grad=True) but no jacobian requested
lightning.qubit
         numpy  qml.np    qml.np    qjit     qjit    qjit
               no-grad      grad    comp  no-grad    grad
  15 -    0.14    0.16       1.3    10.4      0.1    error
  16 -    0.24    0.25       2.0    11.6      0.2 
  17 -    0.44    0.42       3.7    12.8      0.3 
  20 -    3.75    3.74      32.6    19.8      3.4 
> NotImplementedError: Converting dtype('O') to a ctypes type
lightning.kokkos (with 32 threads)
         numpy  qml.np    qml.np    qjit     qjit    qjit
               no-grad      grad    comp  no-grad    grad
  15 -     0.1     0.1       0.7    10.3      0.0        
  20 -     0.3     0.3       2.4    16.6      0.3        
  23 -     1.4     1.4      15.1    21.5      1.3        
  25 -     6.9     6.9     101.1    30.7      7.3        
```

### `lightning-kokkos` from source with CUDA

lightning-kokkos with GPU
- https://pypi.org/project/PennyLane-Lightning-Kokkos/
- https://docs.pennylane.ai/projects/lightning/en/stable/lightning_kokkos/installation.html
- https://github.com/PennyLaneAI/lightning-on-hpc/blob/main/DataCollection/distributed/LUMI_LKOKKOS_VQE/README.md- 

``` bash
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
pip install ray

# pip install pennylane-catalyst  # [added later]

# install lightning-qubit as prerequisite
CXX=$(which CC) python -m pip install -e . --verbose

CXX=$(which CC) CMAKE_ARGS="-DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80:BOOL=ON -DCMAKE_CXX_COMPILER=$(which CC)" PL_BACKEND="lightning_kokkos" python -m pip install . --verbose
```

Start interactive job on GPU node for testing
``` bash
salloc -q interactive -C gpu -t 0:30:00 -A m4693

# and execute in this interactive session:

source /global/common/software/m4693/venv/qml_LK_GPU/bin/activate
cd nersc/

# to restrict the number of threads:
#export OMP_NUM_THREADS=1

python3 single_circuits/demo_variational.py -q lightning.kokkos -n 25
```

Stats on interactive GPU node (nid200381)
```
lightning.kokkos
  23 -    s
  25 -  3 s
  26 -  6 s
  27 - 12 s
  28 - 25 s

> Benchmarking numpy/qml.numpy, gradients
> no-grad: qml.np.array(requires_grad=True) but no jacobian requested
lightning.kokkos
         numpy  qml.np  jacobian    qjit     qjit    qjit
               no-grad      grad    comp  no-grad    grad
  22 -       s     2 s       5 s    20 s      1 s
  23 -       s     4 s       9 s
  25 -     3 s    18 s      37 s    50 s     24 s
  26 -     6 s
  27 -    12 s
  28 -    25 s
> Kokkos::Cuda ERROR: Failed to call Kokkos::Cuda::finalize()
```

Run batch of circuits in parallel
``` bash
# @ray.remote(num_gpus=0.5) has same runtime than num_gpus=1
time python3 single_circuits/batch_variational.py -n 26 -s 4

# move task to background and monitor GPU usage
nvidia-smi
```

Stats on 1 interactive GPU node
```
ray_init in 7 to 15 s
> How long does 1 circuit run on its GPU?
25 features
  samples run_time  run_time/sample*gpu
  -                 3
 16       32        8
26 features
  samples run_time  run_time/sample*gpu
  -                 6
  4       10        10
  8       23        11
 16       39        10
 32       77        10
> create dev 1.8 s
> create circuit < 1 ms
27 features
  samples run_time  run_time/sample*gpu
  -                 12
  4       16        16
  8       31        15
> Overhead of 4 s per circuit with Ray
> This includes creating dev + circuit

30 features
  samples run_time  run_time/sample*gpu
  -                 n.a.
  4       120       120
> create dev 3.3 s
> create circuit < 1 ms

> Run r circuits sequentially within 1 ray job:
batch_variational.py -n 26 -s 32 -r 8
  total: 48.949 s
  per_circuit: 6.119 s
> per circuit runtime is equivalent to run w/o ray
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
