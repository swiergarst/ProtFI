#!/bin/bash
#SBATCH --job-name=olink_impute
#SBATCH --output=olink_impute_%j.log
#SBATCH --error=olink_impute_%j.err
#SBATCH --cpus-per-task=8     # Number of CPU cores
#SBATCH --mem-per-cpu=16G
#SBATCH --ntasks=1

#module load python/3.8  # Adjust based on your HPC environment

source activate olink_aging_lieke  # Activate your virtual environment if needed

# Start memory monitoring in the background
( while true; do free -h >> memory_usage.log; sleep 10; done ) &

# Run the Python script
python ProteinAge.py

# Stop memory monitoring after the script finishes
kill %1
