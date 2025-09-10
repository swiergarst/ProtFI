#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem 64gb

epochs=100
lr=0.00063
wd=0.0001
bs=4060
gamma=0.99
beta=0.1
net="ds_default"
nruns=1
log=1
init="normal"
var_t=0

# shouldn't change these parameters (are in other script
dset="cmb_ffs"
target="mort"

python ../src/run_nn.py --var_t ${var_t} --init ${init} --wd ${wd} --epochs ${epochs} --bs ${bs} --gamma ${gamma} --beta ${beta} --beta ${beta} --dset ${dset} --net ${net} --nruns ${nruns} --target ${target} --log ${log} --lr ${lr}
