#!/usr/bin/env python3
import numpy as np
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
    for arg in vars(args):
        print( 'myArgs:',arg, getattr(args, arg))
    myRank=os.environ['SLURM_PROCID']
    print('I am rank:',myRank)
    with open(args.input, 'r') as file:
        info = json.load(file)
    matrix = np.random.random((info['nrows'],info['ncols']))
    for i in range(info['iterations']):
        matrix += np.random.random((info['nrows'],info['ncols']))
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
