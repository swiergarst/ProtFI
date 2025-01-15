#!/bin/bash


beginseed=1
endseed=100



for (( i=$beginseed; i<=$endseed; i++ ))
do
    sbatch nn_bootstrap.sbatch -s $i
done
