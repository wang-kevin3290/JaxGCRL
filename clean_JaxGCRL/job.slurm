#!/bin/bash
#SBATCH --job-name=JaxGCRL_run            # Job name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=1                # Number of CPU cores per task
#SBATCH --mem-per-cpu=8G                 # Memory per CPU core
#SBATCH --gres=gpu:1                     # Number of GPUs per node
#SBATCH --constraint=gpu80                # GPU type
#SBATCH --time=3:59:00                  # Time limit (hh:mm:ss)
#SBATCH --mail-type=begin                # Send email when job begins
#SBATCH --mail-type=end                  # Send email when job ends
#SBATCH --mail-user=kw6487@princeton.edu  # Your email

# TRAIN_SCRIPT="train_ant_bs512.sh";  --constraint=gpu80                # GPU type

echo "Waiting for _ hour before starting..."
sleep 0  # Sleep for _ seconds 
echo "Starting job execution..."


module purge
module load anaconda3/2024.2
conda activate expl-env

ls -l /scratch/network/kw6487/JaxGCRL/clean_JaxGCRL/train_crl_jax_brax.py
chmod +x /scratch/network/kw6487/JaxGCRL/clean_JaxGCRL/train_crl_jax_brax.py
# python /scratch/network/kw6487/JaxGCRL/clean_JaxGCRL/train_crl_jax_brax.py

cd /scratch/network/kw6487/JaxGCRL/clean_JaxGCRL/
python train_crl_jax_brax.py --env_id "humanoid" --critic_network_width 256 --actor_network_width 256 --batch_size 256 --num_sgd_batches_per_training_step 400 --num_envs 2048 # --env_id "ant" --batch_size 1024 --batchdiv2 2 --actor_batch_size_multiplier 0.5 # --env_id "humanoid" --critic_network_width 256 --actor_network_width 256 --batch_size 1024 --num_sgd_batches_per_training_step 100 --batchdiv2 2 --actor_batch_size_multiplier 0.5 
# chmod +x "./scripts/${TRAIN_SCRIPT}"
# "./scripts/${TRAIN_SCRIPT}"


