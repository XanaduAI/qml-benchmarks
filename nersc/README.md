# Run QML Benchmarks on Perlmutter

## Setup Podman

All the following commands to be executed on Perlmutter.

Build podman image from dockerfile:
```
podman-hpc build -f Dockerfile.ubu22-PennyLane -t tgermain/ubu22-pennylane > podman_build.out
```

**TODO:** Add command to install Ray in dockerfile

... or install in new image
```
podman-hpc run -it --name ray tgermain/ubu22-pennylane
# in the container
pip install ray
exit
# 
podman-hpc commit ray tgermain/ubu22-pennylane-ray
```

Locally install `qml_benchmarks` with dependencies for development:
```
mkdir qml-benchmarks-devel/nersc/local

IMG=tgermain/ubu22-pennylane-ray
CFSH=/global/cfs/cdirs/m4693  # CFS home
REPO_DIR=$CFSH/qml-benchmarks-devel  # qml-benchmark repo
LOCAL_DIR=$REPO_DIR/nersc/local  # to store local python files
WORK_DIR=$REPO_DIR/nersc/ray/workdir  # to store output files

podman-hpc run -it \
    --volume $LOCAL_DIR:/root \
    --volume $REPO_DIR:/qml-benchmarks \
    --volume $WORK_DIR:/work_dir \
    -e HDF5_USE_FILE_LOCKING='FALSE' \
    --workdir /work_dir \
    $IMG bash

# in the container
cd /qml-benchmarks
pip3 install --user .  # install in /root/.local
```

**Note:** `pip3 install --user .` will install in `/root/.local`, mounted to container.

To make image available on CPU/GPU nodes, either:
```
podman-hpc migrate tgermain/ubu22-pennylane-ray[:version]
```
or make available for everyone in project:
```
IMG=tgermain/ubu22-pennylane-ray
POD_PUB=$CFS/m4693/podman/
podman-hpc --squash-dir $POD_PUB migrate $IMG
chmod -R a+rx $POD_PUB   # to allow anyone to use this image
```

**TODO:** Check and update instructions about migrate for project
