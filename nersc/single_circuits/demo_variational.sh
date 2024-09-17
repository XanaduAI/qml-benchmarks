#!/bin/bash -e

#for NQ in 15 16 17 20; do  # lightning.qubit
#for NQ in 15 20 23 25; do  # lightning.kokkos CPU
for NQ in 15 20 22 23 25 26; do  # lightning.kokkos GPU
    #echo $NQ
    python3 demo_variational.py -n $NQ -r $* 2>/dev/null
done
