#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem 16gb

epochs=100
#lr=5e-6
#wd=1e-5
wd=0.01
lr=0.000005
bs=16
gamma=0.99
beta=0.1
net="ds_default"
nruns=1
log=1

# shouldn't change these parameters (are in other script
dset="allprot"
target="frailty"

python ../src/run_nn.py --lr ${lr} --epochs ${epochs} --bs ${bs} --gamma ${gamma} --wd ${wd} --beta ${beta} --dset ${dset} --net ${net} --nruns ${nruns} --target ${target} --log ${log}
