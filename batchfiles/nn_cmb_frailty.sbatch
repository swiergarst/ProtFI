#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem 16gb

epochs=40
#lr=5e
lr=0.00316
#wd=1e-5
wd=0.001
bs=256
nested_batch_size=0
gamma=0.99
beta=0.0325
net="ds_default"
nruns=1
log=1
var_t=0
add_age=1
seed=9
combine_sets=0
large_head=0

# shouldn't change these parameters (are in other script
dset="cmb"
target="ft_groups"

cd ..


python run_nn.py --large_head ${large_head} --combine_sets ${combine_sets} --seed ${seed} --nested_batch_size ${nested_batch_size} --add_age ${add_age} --var_t ${var_t} --lr ${lr} --epochs ${epochs} --bs ${bs} --gamma ${gamma} --wd ${wd} --beta ${beta} --dset ${dset} --net ${net} --nruns ${nruns} --target ${target} --log ${log}
