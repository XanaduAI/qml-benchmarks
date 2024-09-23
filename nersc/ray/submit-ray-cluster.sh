#!/bin/bash
# Script to launch a Ray cluster modified from
# https://github.com/NERSC/slurm-ray-cluster
# Runs the Python script with a dummy Ray experiment related to hyperparameter
# optimization

#SBATCH -C gpu
#SBATCH --time=00:10:00
#SBATCH -A m4693
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=4
#SBATCH --output=run-%j.txt

module load pytorch/2.3.1

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<<"$head_node_ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    head_node_ip=${ADDR[1]}
  else
    head_node_ip=${ADDR[0]}
  fi
  echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "Starting Ray head node on $ip_head..."
srun --nodes=1 --ntasks=1 bash -c "
    export LC_ALL=C.UTF-8
    export LANG=C.UTF-8
    ray start --head --node-ip-address=$head_node_ip --port=$port
    sleep infinity
" &
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 --exclude=$(hostname) bash -c "
    export LC_ALL=C.UTF-8
    export LANG=C.UTF-8
    ray start --address $head_node_ip:$port
    sleep infinity
" &
sleep 5
#########

##############################################################################################
python test_ray.py
exit
