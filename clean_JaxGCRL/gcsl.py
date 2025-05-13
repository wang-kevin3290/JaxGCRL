

#!/usr/bin/env python3
import subprocess



# env_id=$1
# eval_env_id=$2
# depth=$3
# num_epochs=$4
# total_env_steps=$5
# disable_entropy=$6
# last_state_only=$7
# conda_env=$8



depths_list = ["8", "16", "32", "64"]
disable_entropy_list = ["0", "1"]
# last_state_only_list = ["0", "1"]
# disable_entropy_list = ["1"]
last_state_only_list = ["0"]
num_seeds = 2
experiments = [
    ("expl-env", "humanoid", "humanoid", "100000000", "100"),
    ("contrastive_rl", "ant_big_maze", "ant_big_maze_eval", "100000000", "100"),
    ("contrastive_rl", "ant_u4_maze", "ant_u4_maze_eval", "100000000", "100"),
    ("contrastive_rl", "ant_u5_maze", "ant_u5_maze", "400000000", "400"),
    ("contrastive_rl", "ant_hardest_maze", "ant_hardest_maze", "200000000", "200"),
    ("expl-env", "arm_push_easy", "arm_push_easy", "100000000", "100"),
    ("expl-env", "arm_push_hard", "arm_push_hard", "100000000", "100"),
    ("expl-env", "arm_binpick_hard", "arm_binpick_hard", "100000000", "100"),
    ("expl-env", "humanoid_u_maze", "humanoid_u_maze", "400000000", "400"),
    ("expl-env", "humanoid_big_maze", "humanoid_big_maze", "400000000", "400"),
]   

for _ in range(num_seeds):
    for exp in experiments:
        for depth in depths_list:
            for disable_entropy in disable_entropy_list:
                for last_state_only in last_state_only_list:
                    exp_pass = (exp[1], exp[2], depth, exp[4], exp[3], disable_entropy, last_state_only, exp[0])
                    subprocess.run(["sbatch", f"gcsl.slurm", *exp_pass], check=True)