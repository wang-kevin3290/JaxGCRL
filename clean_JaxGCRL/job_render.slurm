#!/bin/bash
#SBATCH --job-name=JaxGCRL_della            # Job name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=1                # Number of CPU cores per task
#SBATCH --mem-per-cpu=8G                 # Memory per CPU core
#SBATCH --gres=gpu:1                     # Number of GPUs per node
#SBATCH --time 01:00:00                 # Time limit (hh:mm:ss)
#SBATCH --constraint=gpu80                # GPU type
#SBATCH --mail-type=begin                # Send email when job begins
#SBATCH --mail-type=end                  # Send email when job ends
#SBATCH --mail-user=kw6487@princeton.edu  # Your email
#SBATCH --output=slurm_logs/slurm-%j.out       # Output file (%j = job ID)

# TRAIN_SCRIPT="train_ant_bs512.sh";  --constraint=gpu80                # GPU type

echo "Waiting for _ hour before starting..."
sleep 0  # Sleep for _ seconds 
echo "Starting job execution..."


module purge
module load anaconda3/2024.2
conda activate expl-env

ls -l /scratch/gpfs/kw6487/JaxGCRL/clean_JaxGCRL/train_crl_jax_brax.py
chmod +x /scratch/gpfs/kw6487/JaxGCRL/clean_JaxGCRL/train_crl_jax_brax.py
# python /scratch/network/kw6487/JaxGCRL/clean_JaxGCRL/train_crl_jax_brax.py

cd /scratch/gpfs/kw6487/JaxGCRL/clean_JaxGCRL
python render.py
# chmod +x "./scripts/${TRAIN_SCRIPT}"
# "./scripts/${TRAIN_SCRIPT}"


