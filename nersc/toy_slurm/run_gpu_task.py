#!/usr/bin/env python3
import numpy as np
import cupy as cp
import json,os 
import time
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input", type=str, help="input file")
    parser.add_argument("--output", type=str, help="output file")
    args = parser.parse_args()

    return args

def main():
    args = parse_args(); 
    myRank=os.environ['SLURM_PROCID']
    print('I am rank:',myRank)

    # Get the number of available CUDA devices
    num_gpus = cp.cuda.runtime.getDeviceCount()
    print("Number of GPUs available: %d" % num_gpus)


    # Access and print some GPU attributes
    for i in range(num_gpus):
        with cp.cuda.Device(i):
            compute_capability = cp.cuda.Device().compute_capability
            total_memory = cp.cuda.Device().mem_info[1]
            print("GPU %d: Compute Capability: %s, Total Memory: %d bytes" % (i, compute_capability, total_memory))

    assert num_gpus ==1  # make sure  there GPU-rank mapping is unique
    
    with open(args.input, 'r') as file:
        info = json.load(file)
    matrix = cp.random.random((info['nrows'],info['ncols']), np.float64)
    for i in range(info['iterations']):
        matrix += cp.random.random((info['nrows'],info['ncols']), np.float64)
        time.sleep(info['pause'])

    output=[]
    for i in range(info['nrows']):
        output.append(np.sum(matrix[i,:]))

    outF=args.output+myRank
    with open(outF, 'w') as file:
        for i in range(len(output)):
            file.write(str(output[i])+"\n")
        file.write("---------------------")

if __name__ == "__main__":
    main()
