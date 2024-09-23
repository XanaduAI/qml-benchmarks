# Ray experiments on SLURM

Ray allows a distributed hyperparameter exploration on a SLURM cluster with the
setup specified in https://github.com/NERSC/slurm-ray-cluster

In this repository, we run a simple Ray experiment within the Python script
`test_ray.py` that is launched by the script `submit-ray-cluster.slurm` with

```
sbatch submit-ray-cluster.slurm
```

The Python script launches several training trails with different combination of
hyperparameters and saves the settings and dummy metrics such as the GPU location
and the train / run time for the trial.
