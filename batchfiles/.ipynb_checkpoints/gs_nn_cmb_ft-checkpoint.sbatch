#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem-per-cpu 16G

pid="4c9v3674"
#pid="0" #use this to start the sweep
epochs=40
#lr=5e-6
#wd=1e-5
bsmin=1000
bsmax=4000
bsinnermin=8
bsinnermax=48
gamma=0.99
beta=0.1
betamin=0
betamax=1
net="ds_default"
nruns=1
wdmin=-3
wdmax=-1
lrmin=-7
lrmax=-5

dset="cmb"
target="ft_mort"

grid_n=1000

python ../src/nn_gridsearch.py --sweep_pid ${pid} --bsinnermin ${bsinnermin} --bsinnermax ${bsinnermax}  --n_workers 1 --betamin ${betamin} --betamax ${betamax} --counts ${grid_n} --wdmin ${wdmin} --wdmax ${wdmax} --lrmin ${lrmin} --lrmax ${lrmax} --epochs ${epochs} --bsmin ${bsmin} --bsmax ${bsmax} --gamma ${gamma} --beta ${beta} --dset ${dset} --net ${net} --nruns ${nruns} --target ${target}
