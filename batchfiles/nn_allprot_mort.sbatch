#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem 64gb

epochs=100
#lr=5e-6
lr=0.000005
wd=0.00001
#wd=1e-5
bs=1000
gamma=0.99
beta=0.1
net="ds_default"
nruns=1
log=1
init="uniform"
var_t=0

# shouldn't change these parameters (are in other script
dset="allprot"
target="mort"

python ../src/run_nn.py --var_t ${var_t} --init ${init} --wd ${wd} --lr ${lr} --epochs ${epochs} --bs ${bs} --gamma ${gamma} --beta ${beta} --beta ${beta} --dset ${dset} --net ${net} --nruns ${nruns} --target ${target} --log ${log}
