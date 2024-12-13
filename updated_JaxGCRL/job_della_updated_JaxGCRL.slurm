#!/bin/bash
#SBATCH --job-name=JaxGCRL_della_updated_JaxGCRL            # Job name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=1                # Number of CPU cores per task
#SBATCH --mem-per-cpu=8G                 # Memory per CPU core
#SBATCH --gres=gpu:1                     # Number of GPUs per node
#SBATCH --time 01:00:00                 # Time limit (hh:mm:ss)
#SBATCH --partition=gpu80                # GPU type
#SBATCH --mail-type=begin                # Send email when job begins
#SBATCH --mail-type=end                  # Send email when job ends
#SBATCH --mail-user=kw6487@princeton.edu  # Your email
#SBATCH --output=slurm_logs/slurm-%j.out       # Output file (%j = job ID)

eval "$(conda shell.bash hook)"
conda activate contrastive_rl

env=ant_ball

for seed in 1 ; do #the seed is overwritten so doesn't matter
  XLA_PYTHON_CLIENT_MEM_FRACTION=.95 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python training.py \
    --project_name test --group_name first_run --exp_name test --num_evals 50 \
    --seed ${seed} --num_timesteps 5000000 --batch_size 256 --num_envs 512 \
    --discounting 0.99 --action_repeat 1 --env_name ${env} \
    --episode_length 1000 --unroll_length 62  --min_replay_size 1000 --max_replay_size 10000 \
    --contrastive_loss_fn infonce_backward --energy_fn l2\
    --multiplier_num_sgd_steps 1 --log_wandb --n_hidden 4 --skip_connections 0
  done

echo "All runs have finished."

