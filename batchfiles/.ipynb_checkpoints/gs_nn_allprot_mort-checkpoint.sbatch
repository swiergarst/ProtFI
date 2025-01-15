#!/bin/bash
#SBATCH -N 1
#SBATCH -c 12
#SBATCH --mem 64gb

epochs=50
#lr=5e-6
#wd=1e-5
bsmin=100
bsmax=2000
gamma=0.99
beta=0.1
net="ds_default"
nruns=1
wdmin=-5
wdmax=-3
lrmin=-5
lrmax=-3

dset="allprot_ffs"
target="mort"

grid_n=300

python ../src/nn_gridsearch.py --n_workers 12 --counts ${grid_n} --wdmin ${wdmin} --wdmax ${wdmax} --lrmin ${lrmin} --lrmax ${lrmax} --epochs ${epochs} --bsmin ${bsmin} --bsmax ${bsmax} --gamma ${gamma} --beta ${beta} --dset ${dset} --net ${net} --nruns ${nruns} --target ${target}
