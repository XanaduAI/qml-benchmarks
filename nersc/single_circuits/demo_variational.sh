#!/bin/bash -e

#for NQ in 15 16 17 20; do
for NQ in 15 20 23 25; do
    #echo $NQ
    python3 demo_variational.py -n $NQ -r $* 2>/dev/null
done
