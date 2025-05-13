import pickle
import os
import jax
import flax
import tyro
import time
import optax
import wandb
import pickle
import random
import wandb_osh
import numpy as np
import flax.linen as nn
import jax.numpy as jnp
import matplotlib.pyplot as plt

from brax import envs
from etils import epath
from dataclasses import dataclass
from collections import namedtuple
from typing import NamedTuple, Any
from wandb_osh.hooks import TriggerWandbSyncHook
from flax.training.train_state import TrainState
from flax.linen.initializers import variance_scaling
from brax.io import html
from brax.io import model

from evaluator import CrlEvaluator
from buffer import TrajectoryUniformSamplingQueue
from memory_bank import MemoryBank, MemoryBankState
@dataclass
class Args:
    exp_name: str = "train" # os.path.basename(__file__)[: -len(".py")]
    seed: int = random.randint(1, 1000) # 16
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "clean_JaxGCRL_test"
    wandb_entity: str = 'wang-kevin3290-princeton-university'
    wandb_mode: str = 'offline'
    wandb_dir: str = '.'
    wandb_group: str = '.'
    capture_vis: bool = True
    vis_length: int = 1000
    checkpoint: bool = True

    #environment specific arguments
    env_id: str = "humanoid" # "ant_push" "ant_hardest_maze" "ant_big_maze" "humanoid" "ant"
    episode_length: int = 1000
    # to be filled in runtime
    obs_dim: int = 0
    goal_start_idx: int = 0
    goal_end_idx: int = 0

    # Algorithm specific arguments
    total_env_steps: int = 100000000 # 50000000
    num_epochs: int = 100 # 50
    num_envs: int = 512
    eval_env_id: str = ""
    num_eval_envs: int = 128
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.99
    logsumexp_penalty_coeff: float = 0.1
    
    #adding in a batch_size_multiplier argument for critic vs. actor batch size
    critic_batch_size_multiplier: float = 1.0 #this has to be less than or equal to 1
    actor_batch_size_multiplier: float = 1.0 #this has to be less than 1

    max_replay_size: int = 10000
    min_replay_size: int = 1000
    
    unroll_length: int  = 62
    
    # ADDING IN A NETWORK WIDTH ARGUMENT
    same_network_width: int = 0
    network_width: int = 256
    critic_network_width: int = 256
    actor_network_width: int = 256
    actor_depth: int = 4
    critic_depth: int = 4
    actor_skip_connections: int = 0 # 0 for no skip connections, >= 0 means the frequency of skip connections (every X layers)
    critic_skip_connections: int = 0 # 0 for no skip connections, >= 0 means the frequency of skip connections (every X layers)
    
    num_episodes_per_env: int = 1 #the number of episodes to sample from each env when sampling data 
    #(to ensure number of batches is consistent as increase batch_size; for now, just a bandaid fix)
    # should be something like batch_size / 256
    training_steps_multiplier: int = 1 #should have the same effect as num_episodes_per_env, hmmm
    use_all_batches: int = 0 # if 1, use all batches; if 0, use a random subset of batches
    num_sgd_batches_per_training_step: int = 800 # this parameter so as to hold the number of batches constant (no matter batch_size, etc)
    
    mrn: int = 0
    memory_bank: int = 0
    memory_bank_size: int = batch_size # this can be modified too
    
    batchdiv2: int = 0 
    # if 1, freeze gradients for second half of batch
    # if 2, split in half along sa and freeze second half of g (Eysenbach ablation, remember it's forward loss)
    #
    # use batch_size * 2 and split in half and freeze gradients and all that (Eysenbach ablation), does not 
    # TODO: if 2, modifies actor such that it uses the half batch size (isolate for critic ablation)
    # can add 3, 4, etc (if diff between 1 and 2, maybe for batch_size ablation we need to have separate for actor and critic)
    # add more for instead of discarding second half, just freeze gradients for second half so symmetric with first
    
    eval_actor: int = 0
    # if 0, use deterministic actor for evaluation
    # if 1, use stochastic actor for evaluation
    # if 2, sample two actions and take the one with the higher Q value
    # if K >= 2, sample K actions and take the one with the highest Q value
    expl_actor: int = 1
    # if 0, use deterministic actor for exploration/collecting data
    # if 1, use stochastic actor for exploration/collecting data
    # if 2, sample two actions and take the one with the higher Q value
    # if K >= 2, sample K actions and take the one with the highest Q value
    
    entropy_param: float = 0.5
    disable_entropy: int = 0
    
    use_relu: int = 0
    
    resnet: str = "noishmistake4_nodense"
    
    num_render: int = 10
    
    
    
    # to be filled in runtime
    env_steps_per_actor_step : int = 0
    """number of env steps per actor step (computed in runtime)"""
    num_prefill_env_steps : int = 0
    """number of env steps to fill the buffer before starting training (computed in runtime)"""
    num_prefill_actor_steps : int = 0
    """number of actor steps to fill the buffer before starting training (computed in runtime)"""
    num_training_steps_per_epoch : int = 0
    """the number of training steps per epoch(computed in runtime)"""

def create_dataset(run_dir, num_trajs):

    # run_dir = "/scratch/gpfs/kw6487/JaxGCRL/clean_JaxGCRL/runs/humanoid_671_20250105-181659"
    # run_dir = "/scratch/gpfs/kw6487/JaxGCRL/clean_JaxGCRL/runs/humanoid_271_20250117-071637" #Humanoid, depth 8 (100k)
    # run_dir = "/scratch/gpfs/kw6487/JaxGCRL/clean_JaxGCRL/runs/humanoid_186_20250105-180537" #Humanoid, depth 32 (100k)
    # run_dir = "/scratch/gpfs/kw6487/JaxGCRL/clean_JaxGCRL/runs/humanoid_758_20250105-190051" #Humanoid, depth 64 (100k)
    args_path = f"{run_dir}/args.pkl"
    params_path = f"{run_dir}/final.pkl"

    eval_env_id = None #if you want to use the env_id in args.eval_env_id, leave this as None

    

    #COPY OVER DEFINITIONS
    

    def make_env(env_id, args):
        print(f"making env with env_id: {env_id}", flush=True)
        if env_id == "reacher":
            from envs.reacher import Reacher
            env = Reacher(
                backend="spring",
            )
            args.obs_dim = 10
            args.goal_start_idx = 4
            args.goal_end_idx = 7
        elif env_id == "pusher":
            from envs.pusher import Pusher
            env = Pusher(
                backend="spring",
            )
            args.obs_dim = 20
            args.goal_start_idx = 10
            args.goal_end_idx = 13
        elif env_id == "ant":
            from envs.ant import Ant
            env = Ant(
                backend="spring",
                exclude_current_positions_from_observation=False,
                terminate_when_unhealthy=True,
            )

            args.obs_dim = 29
            args.goal_start_idx = 0
            args.goal_end_idx = 2

        elif "ant" in env_id and "maze" in env_id: #needed the add the ant check to differentiate with humanoid maze
            if "gen" not in env_id:
                from envs.ant_maze import AntMaze
                env = AntMaze(
                    backend="spring",
                    exclude_current_positions_from_observation=False,
                    terminate_when_unhealthy=True,
                    maze_layout_name=env_id[4:]
                )

                # args.obs_dim = 29
                # args.goal_start_idx = 0
                # args.goal_end_idx = 2
            else:
                from envs.ant_maze_generalization import AntMazeGeneralization
                gen_idx = env_id.find("gen")
                maze_layout_name = env_id[4:gen_idx-1]
                generalization_config = env_id[gen_idx+4:]
                print(f"maze_layout_name: {maze_layout_name}, generalization_config: {generalization_config}", flush=True)
                env = AntMazeGeneralization(
                    backend="spring",
                    exclude_current_positions_from_observation=False,
                    terminate_when_unhealthy=True,
                    maze_layout_name=maze_layout_name,
                    generalization_config=generalization_config
                )

                args.obs_dim = 29
                args.goal_start_idx = 0
                args.goal_end_idx = 2
        
        elif env_id == "ant_ball":
            from envs.ant_ball import AntBall
            env = AntBall(
                backend="spring",
                exclude_current_positions_from_observation=False,
                terminate_when_unhealthy=True,
            )

            args.obs_dim = 31
            args.goal_start_idx = 28
            args.goal_end_idx = 30

        elif env_id == "ant_push":
            from envs.ant_push import AntPush
            env = AntPush(
                backend="mjx",
            )

            args.obs_dim = 31
            args.goal_start_idx = 0
            args.goal_end_idx = 2
            
        elif env_id == "humanoid":
            from envs.humanoid import Humanoid
            env = Humanoid(
                backend="spring",
                exclude_current_positions_from_observation=False,
                terminate_when_unhealthy=True,
            )

            args.obs_dim = 268
            args.goal_start_idx = 0
            args.goal_end_idx = 3
            
        elif "humanoid" in env_id and "maze" in env_id:
            from envs.humanoid_maze import HumanoidMaze
            env = HumanoidMaze(
                backend="spring",
                maze_layout_name=env_id[9:]
            )

            args.obs_dim = 268
            args.goal_start_idx = 0
            args.goal_end_idx = 3

            
        elif env_id == "arm_reach":
            from envs.manipulation.arm_reach import ArmReach
            env = ArmReach(
                backend="mjx",
            )

            args.obs_dim = 13
            args.goal_start_idx = 7
            args.goal_end_idx = 10
            
        elif env_id == "arm_binpick_easy":
            from envs.manipulation.arm_binpick_easy import ArmBinpickEasy
            env = ArmBinpickEasy(
                backend="mjx",
            )

            args.obs_dim = 17
            args.goal_start_idx = 0
            args.goal_end_idx = 3
            
        elif env_id == "arm_binpick_hard":
            from envs.manipulation.arm_binpick_hard import ArmBinpickHard
            env = ArmBinpickHard(
                backend="mjx",
            )

            args.obs_dim = 17
            args.goal_start_idx = 0
            args.goal_end_idx = 3
            
        elif env_id == "arm_binpick_easy_EEF":
            from envs.manipulation.arm_binpick_easy_EEF import ArmBinpickEasyEEF
            env = ArmBinpickEasyEEF(
                backend="mjx",
            )

            args.obs_dim = 11
            args.goal_start_idx = 0
            args.goal_end_idx = 3
        
        elif "arm_grasp" in env_id: # either arm_grasp or arm_grasp_0.5, etc
            from envs.manipulation.arm_grasp import ArmGrasp
            cube_noise_scale = float(env_id[10:]) if len(env_id) > 9 else 0.3
            env = ArmGrasp(
                cube_noise_scale=cube_noise_scale,
                backend="mjx",
            )

            args.obs_dim = 23
            args.goal_start_idx = 16
            args.goal_end_idx = 23
        
        elif env_id == "arm_push_easy":
            from envs.manipulation.arm_push_easy import ArmPushEasy
            env = ArmPushEasy(
                backend="mjx",
            )

            args.obs_dim = 17
            args.goal_start_idx = 0
            args.goal_end_idx = 3
        
        elif env_id == "arm_push_hard":
            from envs.manipulation.arm_push_hard import ArmPushHard
            env = ArmPushHard(
                backend="mjx",
            )

            args.obs_dim = 17
            args.goal_start_idx = 0
            args.goal_end_idx = 3

        else:
            raise NotImplementedError
        
        return env

    lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
    bias_init = nn.initializers.zeros
    def residual_block(x, width, normalize, activation):
        identity = x
        x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)
        x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)
        x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)
        x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)
        x = x + identity
        return x

    class SA_encoder(nn.Module):
        norm_type = "layer_norm"
        network_width: int = 1024
        network_depth: int = 4
        skip_connections: int = 0
        use_relu: int = 0
        @nn.compact
        def __call__(self, s: jnp.ndarray, a: jnp.ndarray):

            lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
            bias_init = nn.initializers.zeros
            
            if self.norm_type == "layer_norm":
                normalize = lambda x: nn.LayerNorm()(x)
            else:
                normalize = lambda x: x
            
            if self.use_relu:
                activation = nn.relu
            else:
                activation = nn.swish
                
            x = jnp.concatenate([s, a], axis=-1)
            #Initial layer
            x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
            x = normalize(x)
            x = activation(x)
            #Residual blocks
            for i in range(self.network_depth // 4):
                x = residual_block(x, self.network_width, normalize, activation)
            #Final layer
            x = nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
            return x
        
    class G_encoder(nn.Module):
        norm_type = "layer_norm"
        network_width: int = 1024
        network_depth: int = 4
        skip_connections: int = 0
        use_relu: int = 0
        @nn.compact
        def __call__(self, g: jnp.ndarray):

            lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
            bias_init = nn.initializers.zeros

            if self.norm_type == "layer_norm":
                normalize = lambda x: nn.LayerNorm()(x)
            else:
                normalize = lambda x: x
            
            if self.use_relu:
                activation = nn.relu
            else:
                activation = nn.swish
            
            x = g
            #Initial layer
            x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
            x = normalize(x)
            x = activation(x)
            #Residual blocks
            for i in range(self.network_depth // 4):
                x = residual_block(x, self.network_width, normalize, activation)
            #Final layer
            x = nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
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
            
            #Initial layer
            x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
            x = normalize(x)
            x = activation(x)
            #Residual blocks
            for i in range(self.network_depth // 4):
                x = residual_block(x, self.network_width, normalize, activation)
            #Final layer
            # x = nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)

            mean = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
            log_std = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
            
            log_std = nn.tanh(log_std)
            log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

            return mean, log_std
        
    import pickle
    with open(args_path, 'rb') as f:
        args = pickle.load(f)

    if eval_env_id:
        env = make_env(eval_env_id, None)
    else:
        env = make_env(args.eval_env_id, args)

    obs_size = env.observation_size
    action_size = env.action_size

    params = model.load_params(params_path)
    alpha_params, actor_params, critic_params = params
    sa_encoder_params, g_encoder_params = critic_params['sa_encoder'], critic_params['g_encoder']
    actor = Actor(action_size=action_size, network_width=args.actor_network_width, network_depth=args.actor_depth, skip_connections=args.actor_skip_connections, use_relu=args.use_relu)
    sa_encoder = SA_encoder(network_width=args.critic_network_width, network_depth=args.critic_depth, skip_connections=args.critic_skip_connections, use_relu=args.use_relu)
    g_encoder = G_encoder(network_width=args.critic_network_width, network_depth=args.critic_depth, skip_connections=args.critic_skip_connections, use_relu=args.use_relu)

    #COLLECT ONE TRAJECTORY

    SEED = 4237
    # Initialize environment
    rng = jax.random.PRNGKey(seed=SEED)
    env_state = jax.jit(env.reset)(rng)

    @jax.jit
    def policy_step(env_state, actor_params):
        means, _ = actor.apply(actor_params, env_state.obs)
        actions = nn.tanh(means)
        return actions

    @jax.jit
    def step(env_state, action, rng):
        next_state = env.step(env_state, action)
        return next_state, next_state

    @jax.jit
    def collect_trajectory(init_state, actor_params, rng, steps=1000):
        def body_fn(i, carry):
            env_state, states, actions, rng = carry
            rng, step_rng = jax.random.split(rng)
            
            # Get action from policy
            action = policy_step(env_state, actor_params)
            
            # Step environment forward
            next_state, _ = step(env_state, action, step_rng)
            
            # Store state and action
            states = states.at[i].set(env_state.obs)
            actions = actions.at[i].set(action)
            
            return next_state, states, actions, rng
        
        # Initialize arrays to store trajectory
        states = jnp.zeros((steps, env.observation_size))
        actions = jnp.zeros((steps, env.action_size))
        
        # Collect trajectory
        final_state, states, actions, _ = jax.lax.fori_loop(
            0, steps, body_fn, (init_state, states, actions, rng)
        )
        
        return states, actions

    NUM_TRAJS = num_trajs

    states_dataset = np.zeros((1000 * NUM_TRAJS, obs_size))
    actions_dataset = np.zeros((1000 * NUM_TRAJS, action_size))

    # Collect trajectory
    for i in range(NUM_TRAJS):
        rng, trajectory_rng = jax.random.split(rng)
        states, actions = collect_trajectory(env_state, actor_params, trajectory_rng)
        states_dataset[i*1000:(i+1)*1000] = states
        actions_dataset[i*1000:(i+1)*1000] = actions