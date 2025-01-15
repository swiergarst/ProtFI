#!/bin/bash
#SBATCH -N 1
#SBATCH -c 12
#SBATCH --mem 64gb

epochs=30
#lr=5e-6
#wd=1e-5
bsmin=100
bsmax=10000
gamma=0.99
beta=0.1
net="ds_default"
nruns=1
wdmin=-6
wdmax=-4
lrmin=-5
lrmax=-2

add_age=1
dset="cmb_met_ffs"
target="mort"

grid_n=300


python ../src/nn_gridsearch.py --n_workers 12 --add_age ${add_age} --counts ${grid_n} --wdmin ${wdmin} --wdmax ${wdmax} --lrmin ${lrmin} --lrmax ${lrmax} --epochs ${epochs} --bsmin ${bsmin} --bsmax ${bsmax} --gamma ${gamma} --beta ${beta} --dset ${dset} --net ${net} --nruns ${nruns} --target ${target}
