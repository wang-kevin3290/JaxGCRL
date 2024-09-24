#!/bin/bash
#SBATCH --job-name=JaxGCRL_run            # Job name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=1                # Number of CPU cores per task
#SBATCH --mem-per-cpu=4G                 # Memory per CPU core
#SBATCH --gres=gpu:1                     # Number of GPUs per node
#SBATCH --time=02:00:00                  # Time limit (hh:mm:ss)
#SBATCH --mail-type=begin                # Send email when job begins
#SBATCH --mail-type=end                  # Send email when job ends
#SBATCH --mail-user=kw6487@princeton.edu  # Your email

TRAIN_SCRIPT="train_ant_bs512.sh"

module purge
module load anaconda3/2024.2
conda activate contrastive_rl

chmod +x "./scripts/${TRAIN_SCRIPT}"
"./scripts/${TRAIN_SCRIPT}"