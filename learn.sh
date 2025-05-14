#!/bin/bash -l

#SBATCH --job-name=learnexample   # create a short name for your job
#SBATCH --partition=pvc9        # cluster partition to be used
#SBATCH --account=support-gpu  # slurm project account
#SBATCH --nodes=1              # number of nodes
# SBATCH --ntasks-per-node=2    # tasks per node
# #SBATCH --cpus-per-task=8      # cpu-cores per task
# #SBATCH --mem=32G              # total memory per node
#SBATCH --gres=gpu:4         # number of allocated gpus per node
# #SBATCH --exclusive            # require exclusive use of node resources
#SBATCH --time=00:10:00        # total run time limit (HH:MM:SS)

source setup.sh
source venv_project/bin/activate

python jaxexample.py

