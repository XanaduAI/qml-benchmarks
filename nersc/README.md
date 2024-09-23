# Experiments on the NERSC cluster

Setup scripts and code to run QML experiments on a SLURM cluster using the 
`qml_benchmarks` package.

# Quick start

The following command can help you start an interactive shell with GPUs
allocated using QOS:

```
salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account XXXX
```

# Running jobs with SBATCH

Create a file with specifications for the resources needed
`submit-job.slurm`.

```
#SBATCH -C gpu
#SBATCH --time=00:10:00
#SBATCH -A m4693
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=4
#SBATCH --output=run-%j.txt

module load pytorch/2.3.1

# Run you code
```

Launch the script:

```
sbatch submit-job.slurm
```
