#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem 64gb

epochs=200
#lr=5e-6
#wd=1e-5
bs=16
gamma=0.99
beta=0.1
net="ds_default"
nruns=1
log=1

# shouldn't change these parameters (are in other script
dset="cmb_met_mh"
target="frailty"

python ../src/run_nn.py --epochs ${epochs} --bs ${bs} --gamma ${gamma} --beta ${beta} --beta ${beta} --dset ${dset} --net ${net} --nruns ${nruns} --target ${target} --log ${log}
