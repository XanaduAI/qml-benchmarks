

## Run with Python `venv`

### `lightning-kokkos` from pypi wheels

Start interactive job on CPU node for testing
``` bash
salloc -q interactive -C cpu -t 0:30:00 -A m4693

# and execute in this interactive session:

source /global/common/software/m4693/venv/qml_LK/bin/activate
cd nersc/

# to restrict the number of threads:
#export OMP_NUM_THREADS=32
python3 -u performance_indicators/perf_ind_kernel.py --numFeatures 4 --inputPath performance_indicators/linearly_separable/
```

Runtimes with Ray on CPU node, dataset with 240x240 = 57,600 kernels
```
> 57600 / 128 = 450 kernels per core
> default.qubit.jax
qubits     4     10     15
real    1m33s  2m42  11m27
user    0m46s  1m00   4m20
sys	    0m22s  0m28   2m21
```