#!/bin/bash
#SBATCH -N 1
#SBATCH -c 12
#SBATCH --mem-per-cpu 16gb


dset="cmb"
target="frailty"
tol=0.001
ranking=1

python forward_selection.py --dset ${dset} --target ${target} --tol ${tol} --ranking ${ranking}
