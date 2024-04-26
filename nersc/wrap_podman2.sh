#!/bin/bash
echo W:myRank is $SLURM_PROCID
IMG=$1
CMD=$2
outPath=$3
CFSH=$4
BASE_DIR=$5
WORK_DIR=$6

if [ $SLURM_PROCID -eq 0 ] ; then 
   echo W:IMG=$IMG 
   echo W:CMD=$CMD
   #echo Q:fire $
fi

echo W:BASE_DIR=$BASE_DIR
echo 'W:start podman'
podman-hpc run -it \
	   --volume $HOME:/home \
	   --volume $CFSH/$BASE_DIR:/$BASE_DIR \
	   --volume $CFSH/$WORK_DIR:$WORK_DIR \
	   -e HDF5_USE_FILE_LOCKING='FALSE' \
	   --workdir $WORK_DIR \
	   $IMG <<EOF 
cd $BASE_DIR
echo P:pwd; pwd
pip3 install .
cd -
$CMD
echo P:done
exit
EOF
  
echo 'W:done podman'
