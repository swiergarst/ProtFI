#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem 64gb

pytest ../src/test_nn.py
