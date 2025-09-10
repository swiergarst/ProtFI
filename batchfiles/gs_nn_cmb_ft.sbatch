#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem-per-cpu 16G

pid="5oi6hz47"
#pid="0" #use this to start the sweep
epochs=100
#lr=5e-6
#wd=1e-5
bsmin=8
bsmax=1000
bsinnermin=8
bsinnermax=48
gamma=0.99
beta=0.1
betamin=0.01
betamax=0.1
net="ds_default"
nruns=1
wdmin=-3
wdmax=-2
lrmin=-6
lrmax=-4

large_head=1
add_age=1
dset="cmb"
target="ft_mort"

grid_n=1000

cd ..

python nn_gridsearch.py --large_head ${large_head} --add_age ${add_age} --sweep_pid ${pid} --bsinnermin ${bsinnermin} --bsinnermax ${bsinnermax}  --n_workers 1 --betamin ${betamin} --betamax ${betamax} --counts ${grid_n} --wdmin ${wdmin} --wdmax ${wdmax} --lrmin ${lrmin} --lrmax ${lrmax} --epochs ${epochs} --bsmin ${bsmin} --bsmax ${bsmax} --gamma ${gamma} --beta ${beta} --dset ${dset} --net ${net} --nruns ${nruns} --target ${target}
