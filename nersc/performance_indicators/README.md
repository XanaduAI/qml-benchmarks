# Instructions for running performance indicators

To get the data for a given model we need to run the python script `perf_ind_variational.py` (for variational models) or 
`perf_ind_kernel.py` (for kernel models). This gets the indicators 'first train step time', 'consec. train step time',
 and 'prediction time'. 

For variational models, there is an additional script `perf_ind_full_train.py` that attempts to run the model until 
convergence in order to get the 'accuracy' and 'train steps for convergence' numbers. Since this can take a long time, it is a separate script. 

The results are saved to a subdirectory specified at the
start of the script (see e.g. JAX/). The script also saves the pip package list for
future reference. 

Within the scripts there are a number of settings that can be changed at the start. This should make it easy to reuse 
the same code when we have updated models. **Please make sure you chose a unique choice of `perf_ind_name` when gathering results for a new
workflow** (i.e. the name of the workflow) since results will be overwritten if the same name is used twice.


The performance indicators use the hyperparameter settings specified in `hyperparam_settings.yaml`. These are chosen to 
be those which require the most compute and shouldn't be changed. 

## determinining the number of CPUs
To avoid wasting resources, you should first determine how many CPUs are required. This can be done by launching an 
interative node:

`salloc -q shared_interactive -C cpu -t 4:00:00 --cpus-per-task=XX`

where you control the number of cpus-per-task. Start with a high number and
monitor the CPU usage with top by SSHing to the compute node in another terminal.

On the interactive node, from the /nersc directory run 

`source pm_podman.source`

to load the container. Then 

`bash installs.sh`

To install the qml-benchmarks package (we also downgrade to a version of scipy that works with catalyst). Run the 
performance indicator pythons script:

`python3 perf_ind_variational.py`

Note the CPU usage in top, and repeat this process (lowering the CPUs each time) until you determine a reasonable
number of CPUs to use, and add this to the `CPUs` row of the google sheet. 

## running a batch job

Once the resources have been decided, edit the file `sbatch submit_job.slr` accordingly and launch a slurm job for 
the performance indicator by running 

`sbatch submit_job.slr`

## recording memory usage
Once the job has finished, run 

`seff JOBID`

where JOBID is the slurm job id. The value `Memory Utilized` is an estimate of the peak memory usage. 
Add this to the sheet. 





