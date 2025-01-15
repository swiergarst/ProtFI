#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem 64gb 


dset="cmb"
target="frailty"
model="trees"
n_trees=1000
max_depth=2
l1_ratio=0.52 # best value for mortality
alpha=0.0021 # best value for mortality
var_t=0.2

python ../src/run_trees.py --var_t ${var_t} --dset ${dset} --target ${target} --n_trees ${n_trees} --model ${model} --max_depth ${max_depth} --l1_ratio ${l1_ratio} --alpha ${alpha} 
