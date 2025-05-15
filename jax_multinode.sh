#!/bin/bash -l
#SBATCH --job-name=jax_distributed_test
#SBATCH --partition=pvc9            # Cluster partition to be used
#SBATCH --account=support-gpu       # slurm project account
#SBATCH --nodes=2                   # Number of nodes
#SBATCH --ntasks-per-node=4         # Number of tasks per node
# #SBATCH --cpus-per-task=8         # CPU cores per task
# #SBATCH --mem=32G                 # Total memory per node
#SBATCH --gres=gpu:4                # GPUs per node
# #SBATCH --exclusive            # require exclusive use of node resources
#SBATCH --time=00:30:00             # Run time limit (HH:MM:SS)
#SBATCH --output=jax_output.log
#SBATCH --error=jax_error.log

# Load Python environment
source setup.sh

# Activate virtual environment
source venv_project/bin/activate

# Set the coordinator node (master) for distributed JAX
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=12345

# Run the distributed JAX script 
srun --mpi=pmi2 python jaxexample.py
