#!/bin/bash

# WARNING: Set GPU_NUM to available GPU on the server in CUDA_VISIBLE_DEVICES=<GPU_NUM>
# or remove this flag entirely if only one GPU is present on the device.

# NOTE: If you run into OOM issues, try reducing --num_envs

eval "$(conda shell.bash hook)"
conda activate contrastive_rl

env=ant
batch_size=1024
num_timesteps=10000000
contrastive_loss_fn=infonce_backward
group_name=${env}

for seed in 1 2 3 4 5 ; do
  XLA_PYTHON_CLIENT_MEM_FRACTION=.95 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python training.py \
    --project_name JaxGCRL_IW --group_name ${group_name} --exp_name ${env}_bs_${batch_size}_${contrastive_loss_fn}_seed${seed} --num_evals 50 \
    --seed ${seed} --num_timesteps ${num_timesteps} --batch_size ${batch_size} --num_envs 512 \
    --discounting 0.99 --action_repeat 1 --env_name ${env} \
    --episode_length 1000 --unroll_length 62  --min_replay_size 1000 --max_replay_size 10000 \
    --contrastive_loss_fn ${contrastive_loss_fn} --energy_fn l2 \
    --multiplier_num_sgd_steps 1 --log_wandb
  done

echo "All runs have finished."