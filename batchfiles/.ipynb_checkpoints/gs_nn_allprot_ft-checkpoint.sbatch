#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem 64gb

epochs=100
#lr=5e-6
#wd=1e-5
bsmin=8
bsmax=1000
gamma=0.99
beta=0.1
net="ds_default"
nruns=1
wdmin=0.00001
wdmax=0.001
lrmin=0.00000005
lrmax=0.0005

dset="allprot"
target="frailty"
grid_n=1000

python ../src/nn_gridsearch.py --counts ${grid_n} --wdmin ${wdmin} --wdmax ${wdmax} --lrmin ${lrmin} --lrmax ${lrmax} --epochs ${epochs} --bsmin ${bsmin} --bsmax ${bsmax} --gamma ${gamma} --beta ${beta} --dset ${dset} --net ${net} --nruns ${nruns} --target ${target}
