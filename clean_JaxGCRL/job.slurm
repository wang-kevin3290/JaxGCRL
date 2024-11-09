#!/bin/bash
#SBATCH --job-name=JaxGCRL_run            # Job name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=1                # Number of CPU cores per task
#SBATCH --mem-per-cpu=8G                 # Memory per CPU core
#SBATCH --gres=gpu:1                     # Number of GPUs per node
#SBATCH --constraint=a100                 # GPU type
#SBATCH --time 4:00:00                 # Time limit (hh:mm:ss)
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

ls -l /scratch/network/kw6487/JaxGCRL/clean_JaxGCRL/train_crl_jax_brax.py
chmod +x /scratch/network/kw6487/JaxGCRL/clean_JaxGCRL/train_crl_jax_brax.py
# python /scratch/network/kw6487/JaxGCRL/clean_JaxGCRL/train_crl_jax_brax.py

cd /scratch/network/kw6487/JaxGCRL/clean_JaxGCRL/
python train_crl_jax_brax.py --env_id "ant_ball" --critic_depth 4 --actor_depth 4 --use_all_batches 1 #--batch_size 512 --num_sgd_batches_per_training_step 800 --num_epochs 10 --total_env_steps 10000000 #--env_id "arm_grasp" --critic_depth 8 --actor_depth 8 --batch_size 512 --num_sgd_batches_per_training_step 800 --num_epochs 50 --total_env_steps 50000000 #--env_id "ant_big_maze" --eval_env_id "ant_big_maze_eval" --critic_depth 6 --actor_depth 6 --critic_skip_connections 0 --actor_skip_connections 0 --num_epochs 100 --total_env_steps 100000000 --batch_size 512 --vis_length 2500 --num_sgd_batches_per_training_step 800 --critic_network_width 256 --actor_network_width 256 #--episode_length 1000 #--max_replay_size 10500 #--num_sgd_batches_per_training_step 800  #--critic_depth 12 --actor_depth 12 #--episode_length 2500 --max_replay_size 25000 #--env_id "arm_grasp" --num_epochs 7 --total_env_steps 700000 #--env_id "humanoid" --critic_depth 20 --actor_depth 20 --actor_skip_connections 5 --critic_skip_connections 5 --critic_network_width 256 --actor_network_width 256 --batch_size 512 --num_sgd_batches_per_training_step 800 #--env_id "arm_binpick_hard" --num_epochs 100 --total_env_steps 100000000 #--env_id "humanoid" --critic_depth 8 --actor_depth 8 --actor_skip_connections 2 --critic_network_width 256 --actor_network_width 256 --batch_size 256 --num_sgd_batches_per_training_step 400 #--env_id "humanoid" --critic_network_width 256 --actor_network_width 256 --batch_size 256 --num_sgd_batches_per_training_step 400 --num_envs 2048 # --env_id "ant" --batch_size 1024 --batchdiv2 2 --actor_batch_size_multiplier 0.5 # --env_id "humanoid" --critic_network_width 256 --actor_network_width 256 --batch_size 1024 --num_sgd_batches_per_training_step 100 --batchdiv2 2 --actor_batch_size_multiplier 0.5 
# chmod +x "./scripts/${TRAIN_SCRIPT}"
# "./scripts/${TRAIN_SCRIPT}"


