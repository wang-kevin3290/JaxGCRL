eval_env_id = 'arm_push_easy'
params_path = '/scratch/gpfs/kw6487/JaxGCRL/clean_JaxGCRL' #'./runs/arm_binpick_easy_224_20241202-013243'
pkl_fname = 'final.pkl'

actor_network_width = 256
actor_network_depth = 32
actor_skip_connections = 0
actor_use_relu = 0

vis_length = 1000
num_rollouts = 10

send_wandb = False
wandb_run_id = "7ukub50o"


def make_env(env_id):
    print(f"making env with env_id: {env_id}", flush=True)
    if env_id == "reacher":
        from envs.reacher import Reacher
        env = Reacher(
            backend="spring",
        )

    elif env_id == "pusher":
        from envs.pusher import Pusher
        env = Pusher(
            backend="spring",
        )

    elif env_id == "ant":
        from envs.ant import Ant
        env = Ant(
            backend="spring",
            exclude_current_positions_from_observation=False,
            terminate_when_unhealthy=True,
        )


    elif "ant" in env_id and "maze" in env_id: #needed the add the ant check to differentiate with humanoid maze
        from envs.ant_maze import AntMaze
        env = AntMaze(
            backend="spring",
            exclude_current_positions_from_observation=False,
            terminate_when_unhealthy=True,
            maze_layout_name=env_id[4:]
        )

    
    elif env_id == "ant_ball":
        from envs.ant_ball import AntBall
        env = AntBall(
            backend="spring",
            exclude_current_positions_from_observation=False,
            terminate_when_unhealthy=True,
        )


    elif env_id == "ant_push":
        from envs.ant_push import AntPush
        env = AntPush(
            backend="mjx",
        )

        
    elif env_id == "humanoid":
        from envs.humanoid import Humanoid
        env = Humanoid(
            backend="spring",
            exclude_current_positions_from_observation=False,
            terminate_when_unhealthy=True,
        )

        
    elif "humanoid" in env_id and "maze" in env_id:
        from envs.humanoid_maze import HumanoidMaze
        env = HumanoidMaze(
            backend="spring",
            maze_layout_name=env_id[9:]
        )


        
    elif env_id == "arm_reach":
        from envs.manipulation.arm_reach import ArmReach
        env = ArmReach(
            backend="mjx",
        )

        
    elif env_id == "arm_binpick_easy":
        from envs.manipulation.arm_binpick_easy import ArmBinpickEasy
        env = ArmBinpickEasy(
            backend="mjx",
        )

        
    elif env_id == "arm_binpick_hard":
        from envs.manipulation.arm_binpick_hard import ArmBinpickHard
        env = ArmBinpickHard(
            backend="mjx",
        )


        
    elif env_id == "arm_binpick_easy_EEF":
        from envs.manipulation.arm_binpick_easy_EEF import ArmBinpickEasyEEF
        env = ArmBinpickEasyEEF(
            backend="mjx",
        )

    
    elif "arm_grasp" in env_id: # either arm_grasp or arm_grasp_0.5, etc
        from envs.manipulation.arm_grasp import ArmGrasp
        cube_noise_scale = float(env_id[10:]) if len(env_id) > 9 else 0.3
        env = ArmGrasp(
            cube_noise_scale=cube_noise_scale,
            backend="mjx",
        )

    
    elif env_id == "arm_push_easy":
        from envs.manipulation.arm_push_easy import ArmPushEasy
        env = ArmPushEasy(
            backend="mjx",
        )

    elif env_id == "arm_push_hard":
        from envs.manipulation.arm_push_hard import ArmPushHard
        env = ArmPushHard(
            backend="mjx",
        )

    else:
        raise NotImplementedError
    
    return env




import jax
from jax import numpy as jp
import matplotlib.pyplot as plt
from IPython.display import HTML
from brax.io import model, html
import matplotlib.pyplot as plt 
import pickle
import numpy as np
import flax.linen as nn
from flax.linen.initializers import variance_scaling
from datetime import datetime
import wandb


env = make_env(eval_env_id)
action_size = env.action_size

lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
bias_init = nn.initializers.zeros
def residual_block(x, width, normalize, activation):
    identity = x
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = x + identity  # Skip connection
    return x

class Actor(nn.Module):
    action_size: int
    norm_type = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    skip_connections: int = 0 # 0 for no skip connections, >= 0 means the frequency of skip connections (every X layers)
    use_relu: int = 0
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    @nn.compact
    def __call__(self, x):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x
            
        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish

        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        
        print(f"x.shape: {x.shape}", flush=True)
        
        #Initial layer
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x) #don't need to normalize or activate because the residual block starts with it
        #Residual blocks
        for i in range(self.network_depth // 2):
            x = residual_block(x, self.network_width, normalize, activation)
        #Final layer
        x = normalize(x)
        x = activation(x)
        mean = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        log_std = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        
        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        
        return mean, log_std
    
actor = Actor(action_size=action_size, network_width=actor_network_width, network_depth=actor_network_depth, skip_connections=actor_skip_connections, use_relu=actor_use_relu)

params = model.load_params(params_path + f'/{pkl_fname}')
_, actor_params, _ = params
# sa_encoder_params, g_encoder_params = encoders_params['sa_encoder'], encoders_params['g_encoder'] #not needed I think, only actor


"""Renders the policy and saves it as an HTML file."""
# JIT compile the rollout function
@jax.jit
def policy_step(env_state, actor_params):
    means, _ = actor.apply(actor_params, env_state.obs)
    actions = nn.tanh(means)
    next_state = env.step(env_state, actions)
    return next_state, env_state  # Return current state for visualization

rollout_states = []
for i in range(num_rollouts):
    print(f"i: {i}", flush=True)
    env = make_env(eval_env_id)
    
    # Initialize environment
    rng = jax.random.PRNGKey(seed=i+1)
    env_state = jax.jit(env.reset)(rng)
    
    # Collect rollout using jitted function
    for j in range(vis_length):
        if j % 10 == 0:
            print(f"j: {j}", flush=True)
        env_state, current_state = policy_step(env_state, actor_params)
        rollout_states.append(current_state.pipeline_state)
        
# try:
#     print(f"len(rollout_states): {len(rollout_states)}", flush=True)
# except Exception as e:
#     print(f"error: {e}", flush=True)

# try:
#     print(f"rollout_states[0].shape: {rollout_states[0].shape}", flush=True)
# except Exception as e:
#     print(f"error: {e}", flush=True)

# try:
#     print(f"rollout_states[0].obs.shape: {rollout_states[0].obs.shape}", flush=True)
# except Exception as e:
#     print(f"error: {e}", flush=True)


# Render and save
html_string = html.render(env.sys, rollout_states)
print(f"len(html_string): {len(html_string)}", flush=True)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
render_path = f"vis/{eval_env_id}__{pkl_fname}_d{actor_network_depth}_s{actor_skip_connections}_{timestamp}.html"
with open(render_path, "w") as f:
    print(f"writing to {render_path}", flush=True)
    f.write(html_string)
# wandb.log({"vis": wandb.Html(html_string)})

if send_wandb:
    wandb.init(id=wandb_run_id, resume=True)
    wandb.log({"vis": wandb.Html(html_string)})
    wandb.finish()

