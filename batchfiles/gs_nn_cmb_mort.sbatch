#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem 64gb

pid=0
#pid="ly7oj6at"
epochs=100
#lr=5e-6
#wd=1e-5
bsmin=1000
bsmax=4000
gamma=0.99
beta=0.1
net="ds_default"
nruns=1
wdmin=-5
wdmax=-2
lrmin=-6
lrmax=-3

add_age=1
dset="cmb"
target="ft_mort"

grid_n=1000

cd ..

python nn_gridsearch.py --sweep_pid ${pid} --n_workers 1 --add_age ${add_age} --counts ${grid_n} --wdmin ${wdmin} --wdmax ${wdmax} --lrmin ${lrmin} --lrmax ${lrmax} --epochs ${epochs} --bsmin ${bsmin} --bsmax ${bsmax} --gamma ${gamma} --beta ${beta} --dset ${dset} --net ${net} --nruns ${nruns} --target ${target}
