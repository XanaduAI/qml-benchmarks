# Instructions for running performance indicators

To get the data for a given model we need to run the python script `perf_ind_variational.py` (for variational models) or 
`perf_ind_kernel.py` (for kernel models). 

Within the scripts there are a number of settings that can be changed at the start. This should make it easy to reuse 
the same code when we have updated models. **Please make sure you chose a unique choice of `perf_ind_name` when gathering results for a new
workflow** (i.e. the name of the workflow) since results will be overwritten if the same name is used twice.

The performance indicators use the hyperparameter settings specified in `hyperparam_settings.yaml`. These are chosen to 
be those which require the most compute and shouldn't be changed. 

To run a performance indicator you need to edit a number of things in the file `nersc/submit_job.slr`:

- Choose the number of features by editing `numFeatures=X`


- Chose whether it is for a variational or kernel model by editing the line

  `CMD=" python3 -u  performance_indicators/XXX.py --numFeatures $numFeatures --inputPath performance_indicators/linearly_separable/ "`

  where `XXX` is either `perf_ind_kernel` or `perf_ind_variational`


- Decide the maximum job time (format HH:MM:SS):

  `#SBATCH -q shared -t 1:00:00`


- Decide the number of CPUs to use (see below):

   `#SBATCH --cpus-per-task=X`

To launch a job run 

`sbatch submit_job.slr`

This will add the job to the queue; when finished the results are stored in `performance_indicators/perf_ind_name`.
Add the jobID to the google sheet for reference. 

## Determinining the number of CPUs
To avoid wasting resources, you should first determine how many CPUs are required. To have an idea of 
CPU usage, launch a job and ssh to the compute node, and run top to see the CPU usage (then kill the job). 
Repeat the process until a reasonable number of CPUs is found (i.e. most are in use). 

Add this choice to the `CPUs` row of the google sheet. 

## Recording memory and CPU usage
Once the job has finished, run 

`seff JOBID`

where JOBID is the slurm job id. Add the corresponding info to the google sheet. 





