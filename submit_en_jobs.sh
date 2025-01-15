#!/bin/bash

# Define arrays for datasets and targets
datasets=('allprot')
targets=('frailty')

# Function to check if a job with a given name is running
is_job_running() {
  local job_name=$1
  # Check the job queue for jobs with the given name
  if squeue --job --format=%j | grep -q "$job_name"; then
    return 0 # Job is running
  else
    return 1 # Job is not running
  fi
}

# Loop through each combination of dataset and target
for dset in "${datasets[@]}"; do
  for target in "${targets[@]}"; do
    # Create a job name based on the combination
    job_name="model_${dset}_${target}"
    
    # Check if the job is already running
    if is_job_running "$job_name"; then
      echo "Job '$job_name' is already running. Skipping submission."
    else
      # Submit the job using sbatch
      sbatch --job-name=$job_name \
             --output=output_linear/output_logs/${job_name}.out \
             --error=output_linear/output_logs/${job_name}.err \
             --wrap="python model_functions.py '$dset' '$target' --combine"
      echo "Submitted job '$job_name'."
    fi
  done
done

