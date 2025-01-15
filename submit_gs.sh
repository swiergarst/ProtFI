#!/bin/bash

new_pid=0

for ((i = 0; i<= 10; i++))
do 
  sbatch gs_nn_cmb_ft.sbatch
done

exit
