#!/bin/bash

#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem 16gb


epochs=5
lr=0.00001
wd=0.001
bs=2000
gamma=0.99
net="ds_default"
log=1
combine_sets=1

dset="cmb"
target="mort"

bootstrap=1

while true;
do
    case "$1" in
        -s) seed=$2; shift 2;;
        --) shift; break ;;
        *) break ;;
    esac
done



cd ..


python run_nn.py --epochs ${epochs} --lr ${lr} --wd ${wd} --bs ${bs} --gamma ${gamma} --net ${net} --log ${log} --combine_sets ${combine_sets} --dset ${dset} --target ${target} --bootstrap ${bootstrap} --seed ${seed}