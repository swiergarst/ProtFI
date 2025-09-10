#!/bin/bash
#SBATCH --job-name=forward_model  # Job name
#SBATCH --output=slurm_%j.out     # Output log (with job ID)
#SBATCH --error=slurm_%j.err      # Error log (with job ID)
#SBATCH --ntasks=1                # Number of tasks (1 CPU task)
#SBATCH --cpus-per-task=10        # Number of CPU cores per task (for n_jobs=10 in the script)
#SBATCH --mem=32G                 # Memory allocation (adjust based on dataset size)


# Load any necessary modules (e.g., Python, scikit-learn, etc.)
module load python/3.12.1  # Adjust the Python version based on your environment

# Activate virtual environment if necessary
# source /path/to/your/venv/bin/activate

# Run the Python script
python ForwardSelection.py

