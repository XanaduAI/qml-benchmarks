#!/bin/bash
echo W:myRank is $SLURM_PROCID
IMG=$1
CMD=$2
RAY_ADDRESS=$3

if [[ $SLURM_PROCID -eq 0 ]]; then 
   echo W:IMG=$IMG 
   echo W:CMD=$CMD
fi

CFSH=/global/cfs/cdirs/m4693  # CFS home
REPO_DIR=$CFSH/qml-benchmarks-devel  # qml-benchmark repo
ROOT_DIR=$REPO_DIR/nersc/local  # to access local python packages
WORK_DIR=$REPO_DIR/nersc  # to store output files

# Mount /tmp to avoid following error with Ray:
#     ValueError: Can't find a `node_ip_address.json` file

PORT=6379

# Script will run in the workdir mounted in the container,
# this will allow us to access the output files easily.

podman-hpc run -it \
    --net host \
    -p $PORT:$PORT \
    --volume /tmp:/tmp \
    --volume $ROOT_DIR:/root \
    --volume $REPO_DIR:/qml-benchmarks \
    --volume $WORK_DIR:/work_dir \
    --workdir /work_dir \
    -e HDF5_USE_FILE_LOCKING='FALSE' \
    --shm-size=10.24gb \
    $IMG <<EOF 
echo P:starting
echo P:PWD=$PWD
ls -l
export RAY_DEDUP_LOGS=0
export RAY_ADDRESS=$RAY_ADDRESS
echo P:RAY_ADDRESS=\${RAY_ADDRESS}
$CMD
echo P:done
exit
EOF
  
echo 'W:done'
