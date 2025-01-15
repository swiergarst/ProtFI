#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem 64gb

epochs=40
lr=0.0000464
wd=0.01
bs=8
gamma=0.99
beta=0.77778
net="ds_default"
nruns=1
log=1
init="normal"
var_t=0
add_age=1
seed=9
combine_sets=0
nested_batch_size=0

# shouldn't change these parameters (are in other script
dset="cmb"
target="ft_mort"


cd ..

python run_nn.py --nested_batch_size ${nested_batch_size} --add_age ${add_age} --seed ${seed} --combine_sets ${combine_sets} --var_t ${var_t} --init ${init} --wd ${wd} --epochs ${epochs} --bs ${bs} --gamma ${gamma} --beta ${beta} --beta ${beta} --dset ${dset} --net ${net} --nruns ${nruns} --target ${target} --log ${log} --lr ${lr}
