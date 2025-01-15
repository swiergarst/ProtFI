#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem-per-cpu 16G

pid="5oy9lds5"
#pid="0" #use this to start the sweep
epochs=40
#lr=5e-6
#wd=1e-5
bsmin=8
bsmax=4000
bsinnermin=8
bsinnermax=48
gamma=0.99
beta=0.1
betamin=0
betamax=1
net="ds_default"
nruns=1
wdmin=-4
wdmax=-2
lrmin=-6
lrmax=-3

add_age=1
dset="cmb"
target="ft_mort"

grid_n=10000

cd ..

python nn_gridsearch.py --add_age ${add_age} --sweep_pid ${pid} --bsinnermin ${bsinnermin} --bsinnermax ${bsinnermax}  --n_workers 1 --betamin ${betamin} --betamax ${betamax} --counts ${grid_n} --wdmin ${wdmin} --wdmax ${wdmax} --lrmin ${lrmin} --lrmax ${lrmax} --epochs ${epochs} --bsmin ${bsmin} --bsmax ${bsmax} --gamma ${gamma} --beta ${beta} --dset ${dset} --net ${net} --nruns ${nruns} --target ${target}
