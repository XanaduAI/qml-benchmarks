#!/bin/bash
echo W:myRank is $SLURM_PROCID
IMG=$1
CMD=$2
CFSH=$3
BASE_DIR=$4
WORK_DIR=$5

if [ $SLURM_PROCID -eq 0 ] ; then 
   echo W:IMG=$IMG 
   echo W:CMD=$CMD
   #echo Q:fire $
fi

echo W:BASE_DIR=$BASE_DIR
echo 'W:start podman'
podman-hpc run -it \
    --volume $CFSH/$BASE_DIR:/root \
    --volume $CFSH/$BASE_DIR:/qml-benchmarks \
    --volume $CFSH/$BASE_DIR/nersc/performance_indicators/linearly_separable:/linearly_separable \
    --volume $CFSH/$WORK_DIR:/qml-benchmarks/nersc \
    -e HDF5_USE_FILE_LOCKING='FALSE' \
    --workdir /qml-benchmarks/nersc \
    $IMG <<EOF 
echo P:pwd; pwd
echo P:all
ls -l ../..
export PYTHONPATH=/qml-benchmarks/src/
$CMD
echo P:done
exit
EOF
  
echo 'W:done podman'

# spare
# 	   --volume $HOME:/home \
    # --volume  $CFSH/qml-benchmarks:/root 
