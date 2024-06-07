#!/bin/bash

echo W:myRank is $SLURM_PROCID

IMG=$1
CMD=$2

if [ $SLURM_PROCID -eq 0 ] ; then 
   echo W:IMG=$IMG 
   echo W:CMD=$CMD
fi

CFSH=/global/cfs/cdirs/m4693  # CFS home
REPO_DIR=$CFSH/qml-benchmarks-devel  # qml-benchmark repo
ROOT_DIR=$REPO_DIR/nersc/local  # to store local python files
WORK_DIR=$REPO_DIR/nersc/ray/workdir  # to store output files

# Script will run in the workdir mounted in the container,
# this will allow us to access the output files easily.

podman-hpc run -it \
    --volume $REPO_DIR:/qml-benchmarks \
    --volume $ROOT_DIR:/root \
    --volume $WORK_DIR:/work_dir \
    -e HDF5_USE_FILE_LOCKING='FALSE' \
    --workdir /work_dir \
    $IMG <<EOF 
echo P:starting
echo P:PWD=$PWD
ls -l
$CMD
echo P:done
exit
EOF
  
echo 'W:done'
