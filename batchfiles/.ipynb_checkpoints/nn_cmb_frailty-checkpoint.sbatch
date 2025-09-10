#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem 16gb

epochs=40
#lr=5e
lr=0.00005
#wd=1e-5
wd=0.005
bs=200
nested_batch_size=8
gamma=0.99
beta=0.75
net="ds_default"
nruns=1
log=1
var_t=0
add_age=1

# shouldn't change these parameters (are in other script
dset="cmb"
target="ft_mort"

python ../src/run_nn.py --nested_batch_size ${nested_batch_size} --add_age ${add_age} --var_t ${var_t} --lr ${lr} --epochs ${epochs} --bs ${bs} --gamma ${gamma} --wd ${wd} --beta ${beta} --dset ${dset} --net ${net} --nruns ${nruns} --target ${target} --log ${log}
