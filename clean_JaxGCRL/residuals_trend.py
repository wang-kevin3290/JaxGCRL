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
import matplotlib.pyplot as plt  # Add matplotlib for plotting

from brax import envs
from etils import epath
from dataclasses import dataclass
from collections import namedtuple
from typing import NamedTuple, Any, Tuple, List
from wandb_osh.hooks import TriggerWandbSyncHook
from flax.training.train_state import TrainState
from flax.linen.initializers import variance_scaling
from brax.io import html

from evaluator import CrlEvaluator
from buffer import TrajectoryUniformSamplingQueue
from memory_bank import MemoryBank, MemoryBankState

from pathlib import Path
import glob

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
    
    save_buffer: int = 0
    
    
    #INSTRUCTIONS TO RUN CHECKPOINT CONTINUATION:
    #-replay_buffer not needed, just need prev to have args.pkl, final.pkl
    #all you need to do is to add --load_prev_ckpt 1 and --prev_slurm_id <prev_slurm_id>
    #-currently, it's set up that the previous run must have set wandb_run_id, env_steps, etc (so you can't continue an old run, they also must be run via this ckpt script)
    load_prev_ckpt: int = 0 #set to 1 if this is a second/third/etc run and need to load previous checkpoint
    prev_slurm_id: str = 0 #prev slurm id to reference to prev slurm's log file (will use this to find the prev's seed and wandb run id); set to 0 if this is the first run
    
    #These will be automatically set/filled in runtime
    wandb_run_id: str = None #will be set as the randomly generated wandb run id for the first, prev's id for second/third/etc
    current_epoch: int = None #will be instantiated to 0 if loading a previous checkpoint, the prev's for second/third/etc; then every epoch it's incremented by 1
    training_state_env_steps: int = None #will be set at the end of training for first, start at prev's for second/third/etc
    training_state_gradient_steps: int = None #will be set at the end of training for first, start at prev's for second/third/etc
    
    
    
    
    # to be filled in runtime
    env_steps_per_actor_step : int = 0
    """number of env steps per actor step (computed in runtime)"""
    num_prefill_env_steps : int = 0
    """number of env steps to fill the buffer before starting training (computed in runtime)"""
    num_prefill_actor_steps : int = 0
    """number of actor steps to fill the buffer before starting training (computed in runtime)"""
    num_training_steps_per_epoch : int = 0
    """the number of training steps per epoch(computed in runtime)"""

lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
bias_init = nn.initializers.zeros

# Modified residual_block to track residual magnitudes
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
    
    # Compute residual before adding
    pre_residual = x
    
    # Add the residual connection
    x = x + identity
    
    # Compute the magnitude of the residual (L2 norm)
    residual_magnitude = jnp.sqrt(jnp.mean(jnp.square(pre_residual), axis=-1))
    
    return x, residual_magnitude

class SA_encoder(nn.Module):
    norm_type = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    skip_connections: int = 0
    use_relu: int = 0
    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray, track_residuals=False):

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
        
        residual_magnitudes = []
        x = jnp.concatenate([s, a], axis=-1)
        #Initial layer
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)
        #Residual blocks
        for i in range(self.network_depth // 4):
            x, res_mag = residual_block(x, self.network_width, normalize, activation)
            if track_residuals:
                residual_magnitudes.append(res_mag)
        #Final layer
        x = nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        if track_residuals:
            return x, residual_magnitudes
        else:
            return x
    
class G_encoder(nn.Module):
    norm_type = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    skip_connections: int = 0
    use_relu: int = 0
    @nn.compact
    def __call__(self, g: jnp.ndarray, track_residuals=False):

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

        residual_magnitudes = []
        x = g
        #Initial layer
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)
        #Residual blocks
        for i in range(self.network_depth // 4):
            x, res_mag = residual_block(x, self.network_width, normalize, activation)
            if track_residuals:
                residual_magnitudes.append(res_mag)
        #Final layer
        x = nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        if track_residuals:
            return x, residual_magnitudes
        else:
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
    def __call__(self, x, track_residuals=False):
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
        
        residual_magnitudes = []
        
        print(f"x.shape: {x.shape}", flush=True)

        # Initial layer
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)
        
        # Residual blocks
        for i in range(self.network_depth // 4):
            x, res_mag = residual_block(x, self.network_width, normalize, activation)
            if track_residuals:
                residual_magnitudes.append(res_mag)
        
        # Final layer
        mean = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        log_std = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        
        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        if track_residuals:
            return mean, log_std, residual_magnitudes
        else:
            return mean, log_std


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner"""
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState
    memory_bank_state: MemoryBankState

class Transition(NamedTuple):
    """Container for a transition"""
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: jnp.ndarray = ()

def load_params(path: str):
    with epath.Path(path).open('rb') as fin:
        buf = fin.read()
    return pickle.loads(buf)

def save_params(path: str, params: Any):
    """Saves parameters in flax format."""
    with epath.Path(path).open('wb') as fout:
        fout.write(pickle.dumps(params))
        

# Function to analyze residual magnitudes
def analyze_actor_residuals(actor, params, obs, batch_size=256):
    """Analyze the magnitudes of residuals in the actor network"""
    # Create a batch of observations (either use real observations or random ones)
    if len(obs.shape) == 2:
        # If obs is already a batch, use it directly or a subset
        batch_obs = obs[:batch_size]
    else:
        # If obs is a single observation, replicate it into a batch
        batch_obs = jnp.tile(obs, (batch_size, 1))
    
    # Forward pass with residual tracking enabled
    _, _, residual_magnitudes = actor.apply(params, batch_obs, track_residuals=True)
    
    # Convert to numpy for easier handling
    residual_magnitudes_np = [jnp.mean(mag).item() for mag in residual_magnitudes]
    
    # Plot the residual magnitudes
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(residual_magnitudes_np)), residual_magnitudes_np)
    plt.title('Magnitude of Residuals Across Actor Network Layers')
    plt.xlabel('Residual Block Index')
    plt.ylabel('Average Residual Magnitude (L2 Norm)')
    plt.grid(True)
    
    # Save the plot
    plt.savefig(f'repeat_reps_figs/{args.env_id}_{args.seed}_actor_residual_magnitudes_{args.actor_depth}.png')
    print(f"Saved: {args.env_id}_{args.seed}_actor_residual_magnitudes_{args.actor_depth}.png", flush=True)
    
    # Also save the data for further analysis
    with open(f'repeat_reps_figs/{args.env_id}_{args.seed}_actor_residual_data_{args.actor_depth}.pkl', 'wb') as f:
        pickle.dump(residual_magnitudes_np, f)
    
    return residual_magnitudes_np
    
# Function to collect trajectory data
def collect_trajectory_data(actor, params, env, env_state, num_steps=100):
    """Collect observation data from a trajectory"""
    trajectory_obs = []
    
    # Store initial observation
    trajectory_obs.append(env_state.obs)
    
    # Run the environment for num_steps
    for _ in range(num_steps):
        # Get action from the actor
        means, _ = actor.apply(params, env_state.obs)
        actions = nn.tanh(means)
        
        # Step the environment
        env_state = env.step(env_state, actions)
        
        # Store observation
        trajectory_obs.append(env_state.obs)
        
        # If done, reset the environment
        if jnp.any(env_state.done):
            # For simplicity, we'll just continue with the states we have if any environment is done
            break
    
    # Stack all observations
    return jnp.concatenate(trajectory_obs, axis=0)

# Modified function to analyze residuals on trajectory data
def analyze_trajectory_residuals(actor, params, env, env_state, num_envs=10, steps_per_env=20, batch_size=256):
    """Analyze residuals on a variety of states from multiple trajectories"""
    # Split the trajectory collection across multiple environments
    env_states = jax.tree_util.tree_map(lambda x: x[:num_envs], env_state)
    
    # Collect trajectory data
    trajectory_obs = collect_trajectory_data(actor, params, env, env_states, num_steps=steps_per_env)
    
    print(f"Collected {trajectory_obs.shape[0]} observations from trajectories", flush=True)
    
    # Randomly sample from trajectory data to create analysis batch
    indices = jax.random.randint(
        jax.random.PRNGKey(0), 
        shape=(batch_size,), 
        minval=0, 
        maxval=trajectory_obs.shape[0]
    )
    sampled_obs = trajectory_obs[indices]
    
    # Forward pass with residual tracking enabled
    _, _, residual_magnitudes = actor.apply(params, sampled_obs, track_residuals=True)
    
    # Convert to numpy for easier handling
    residual_magnitudes_np = [jnp.mean(mag).item() for mag in residual_magnitudes]
    
    # Plot the residual magnitudes
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(residual_magnitudes_np)), residual_magnitudes_np)
    plt.title('Magnitude of Residuals Across Actor Network Layers')
    plt.xlabel('Residual Block Index')
    plt.ylabel('Average Residual Magnitude (L2 Norm)')
    plt.grid(True)
    
    # Save the plot
    plt.savefig('trajectory_residual_magnitudes.png')
    print(f"Saved: trajectory_residual_magnitudes.png", flush=True)
    
    # Also save the data for further analysis
    with open('trajectory_residual_data.pkl', 'wb') as f:
        pickle.dump(residual_magnitudes_np, f)
    
    return residual_magnitudes_np

# Function to analyze SA_encoder residuals
def analyze_sa_encoder_residuals(sa_encoder, params, obs, actions, batch_size=256):
    """Analyze the magnitudes of residuals in the SA encoder network"""
    # Create a batch of observations and actions, excluding last 3 elements from obs
    if len(obs.shape) == 2:
        batch_obs = obs[:batch_size, :-3]  # Modified to exclude last 3 elements
        batch_actions = actions[:batch_size] if len(actions.shape) == 2 else jnp.tile(actions, (batch_size, 1))
    else:
        batch_obs = jnp.tile(obs[:-3], (batch_size, 1))  # Modified to exclude last 3 elements
        batch_actions = jnp.tile(actions, (batch_size, 1))
    
    # Forward pass with residual tracking enabled
    _, residual_magnitudes = sa_encoder.apply(params, batch_obs, batch_actions, track_residuals=True)
    
    # Convert to numpy for easier handling
    residual_magnitudes_np = [jnp.mean(mag).item() for mag in residual_magnitudes]
    
    return residual_magnitudes_np

# Function to analyze G_encoder residuals
def analyze_g_encoder_residuals(g_encoder, params, obs, batch_size=256):
    """Analyze the magnitudes of residuals in the G encoder network"""
    # Create a batch of goals (last 3 elements of obs)
    if len(obs.shape) == 2:
        batch_goals = obs[:batch_size, -3:]  # Take last 3 elements as goals
    else:
        batch_goals = jnp.tile(obs[-3:], (batch_size, 1))  # Take last 3 elements as goals
    
    # Forward pass with residual tracking enabled
    _, residual_magnitudes = g_encoder.apply(params, batch_goals, track_residuals=True)
    
    # Convert to numpy for easier handling
    residual_magnitudes_np = [jnp.mean(mag).item() for mag in residual_magnitudes]
    
    return residual_magnitudes_np

# Modified function to analyze all networks on trajectory data
def analyze_all_networks_residuals(actor, actor_params, sa_encoder, sa_params, g_encoder, g_params, 
                                 env, env_state, args, num_envs=10, steps_per_env=20, batch_size=256):
    """Analyze residuals on all networks using trajectory data"""
    # Split the trajectory collection across multiple environments
    env_states = jax.tree_util.tree_map(lambda x: x[:num_envs], env_state)
    
    # Collect trajectory data
    trajectory_obs = collect_trajectory_data(actor, actor_params, env, env_states, num_steps=steps_per_env)
    
    print(f"Collected {trajectory_obs.shape[0]} observations from trajectories", flush=True)
    
    # Sample random indices for batch creation
    indices = jax.random.randint(
        jax.random.PRNGKey(0), 
        shape=(batch_size,), 
        minval=0, 
        maxval=trajectory_obs.shape[0]
    )
    sampled_obs = trajectory_obs[indices]
    
    # Generate actions for SA encoder
    means, _ = actor.apply(actor_params, sampled_obs)
    actions = nn.tanh(means)
    
    # Get residuals from all networks
    actor_residuals = analyze_actor_residuals(actor, actor_params, sampled_obs)
    sa_residuals = analyze_sa_encoder_residuals(sa_encoder, sa_params, sampled_obs, actions)
    g_residuals = analyze_g_encoder_residuals(g_encoder, g_params, sampled_obs)  # Now passing full obs
    
    # Create comparison plot
    plt.figure(figsize=(15, 8))
    
    plt.subplot(1, 2, 1)  # First subplot for individual plots
    plt.plot(range(len(actor_residuals)), actor_residuals, 'o-', label='Actor')
    plt.plot(range(len(sa_residuals)), sa_residuals, 's-', label='SA Encoder')
    plt.plot(range(len(g_residuals)), g_residuals, '^-', label='G Encoder')
    plt.title('Residual Magnitudes Across Networks')
    plt.xlabel('Residual Block Index')
    plt.ylabel('Average Residual Magnitude (L2 Norm)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)  # Second subplot for boxplot comparison
    data = [actor_residuals, sa_residuals, g_residuals]
    plt.boxplot(data, labels=['Actor', 'SA Encoder', 'G Encoder'])
    plt.title('Distribution of Residual Magnitudes')
    plt.ylabel('Residual Magnitude (L2 Norm)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'repeat_reps_figs/{args.env_id}_{args.seed}_all_networks_residuals.png')
    print(f"Saved: {args.env_id}_{args.seed}_all_networks_residuals.png", flush=True)
    
    # Save the raw data
    residual_data = {
        'actor': actor_residuals,
        'sa_encoder': sa_residuals,
        'g_encoder': g_residuals
    }
    with open(f'repeat_reps_figs/{args.env_id}_{args.seed}_all_networks_residual_data.pkl', 'wb') as f:
        pickle.dump(residual_data, f)
    
    return actor_residuals, sa_residuals, g_residuals

if __name__ == "__main__":   

    prev_run_folder = "/scratch/gpfs/kw6487/JaxGCRL/clean_JaxGCRL/runs/humanoid_271_20250117-071637" #DEPTH 8
    prev_run_folder = "/scratch/gpfs/kw6487/JaxGCRL/clean_JaxGCRL/runs/humanoid_186_20250105-180537" #DEPTH 32
    # prev_run_folder = "/scratch/gpfs/kw6487/JaxGCRL/clean_JaxGCRL/runs/humanoid_758_20250105-190051" #DEPTH 64
    print(f"prev_run_folder: {prev_run_folder}", flush=True)
    
    prev_args_path = Path(prev_run_folder) / "args.pkl"
    prev_params_path = Path(prev_run_folder) / "final.pkl"
    
    import pickle
    with open(prev_args_path, 'rb') as f:
        args = pickle.load(f)   

    # Print every arg
    PRINT_ARGS = 0
    if PRINT_ARGS:
        print("Arguments:", flush=True)
        for arg, value in vars(args).items():
            print(f"{arg}: {value}", flush=True)
        print("\n", flush=True)

    key = jax.random.PRNGKey(args.seed)
    key, buffer_key, env_key, eval_env_key, actor_key, sa_key, g_key, sym_key, asym_key, memory_bank_key = jax.random.split(key, 10)

    def make_env(env_id=args.env_id):
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

                args.obs_dim = 29
                args.goal_start_idx = 0
                args.goal_end_idx = 2
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
        
    env = make_env()
    env = envs.training.wrap(
        env,
        episode_length=args.episode_length,
    )

    obs_size = env.observation_size
    action_size = env.action_size
    env_keys = jax.random.split(env_key, args.num_envs)
    env_state = jax.jit(env.reset)(env_keys)
    env.step = jax.jit(env.step)
    
    print(f"obs_size: {obs_size}, action_size: {action_size}", flush=True)
    
    
    if not args.eval_env_id:
        args.eval_env_id = args.env_id
        
    # make eval env
    eval_env = make_env(args.eval_env_id)
    eval_env = envs.training.wrap(
        eval_env,
        episode_length=args.episode_length,
    )
    eval_env_keys = jax.random.split(eval_env_key, args.num_envs)
    eval_env_state = jax.jit(eval_env.reset)(eval_env_keys)
    eval_env.step = jax.jit(eval_env.step)
        
    
    
    # Network setup
    # Actor
    actor = Actor(action_size=action_size, network_width=args.actor_network_width, network_depth=args.actor_depth, skip_connections=args.actor_skip_connections, use_relu=args.use_relu)
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, np.ones([1, obs_size])),
        tx=optax.adam(learning_rate=args.actor_lr)
    )

    # Critic
    sa_encoder = SA_encoder(network_width=args.critic_network_width, network_depth=args.critic_depth, skip_connections=args.critic_skip_connections, use_relu=args.use_relu)
    sa_encoder_params = sa_encoder.init(sa_key, np.ones([1, args.obs_dim]), np.ones([1, action_size]))
    g_encoder = G_encoder(network_width=args.critic_network_width, network_depth=args.critic_depth, skip_connections=args.critic_skip_connections, use_relu=args.use_relu)
    g_encoder_params = g_encoder.init(g_key, np.ones([1, args.goal_end_idx - args.goal_start_idx]))
        
    critic_state = TrainState.create(
        apply_fn=None,
        params={
            "sa_encoder": sa_encoder_params, 
            "g_encoder": g_encoder_params
            },
        tx=optax.adam(learning_rate=args.critic_lr),
    )

    # Entropy coefficient
    target_entropy = -args.entropy_param * action_size # action_size = 8 for ant, 17 for humanoid, etc # USEED TO BE -0.5 * action_size
    log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
    alpha_state = TrainState.create(
        apply_fn=None,
        params={"log_alpha": log_alpha},
        tx=optax.adam(learning_rate=args.alpha_lr),
    )
        
    
    def jit_wrap(memory_bank):
        memory_bank.insert = jax.jit(memory_bank.insert)
        memory_bank.sample = jax.jit(memory_bank.sample)
        return memory_bank
    
    if args.memory_bank:
        memory_bank = jit_wrap(MemoryBank(memory_bank_size=args.memory_bank_size, feature_dim=64, batch_size=args.batch_size))
        memory_bank_state = jax.jit(memory_bank.init)(memory_bank_key)
    else:
        memory_bank_state = None
    
    # Trainstate
    training_state = TrainingState(
        env_steps=jnp.zeros(()),
        gradient_steps=jnp.zeros(()),
        actor_state=actor_state,
        critic_state=critic_state,
        alpha_state=alpha_state,
        memory_bank_state=memory_bank_state,
    )
    
    # if args.load_prev_ckpt:
    prev_args = pickle.load(open(prev_args_path, "rb"))
    training_state = training_state.replace(
        env_steps=prev_args.training_state_env_steps,
        gradient_steps=prev_args.training_state_gradient_steps,
    )
    
    # If continuing from a previous run, load the saved parameters and OVERWRITE the initial parameters
    # if args.load_prev_ckpt:        
    from brax.io import model
    try:
        params = model.load_params(prev_params_path)
        alpha_params, actor_params, critic_params = params
        # sa_encoder_params, g_encoder_params = critic_params['sa_encoder'], critic_params['g_encoder']
        print(f"Loaded alpha, actor, and critic params from {prev_params_path}", flush=True)
    except:
        print(f"Failed to load params from {prev_params_path}", flush=True)
        
    # replace the initial parameters with the loaded ones
    alpha_state = alpha_state.replace(params=alpha_params)
    actor_state = actor_state.replace(params=actor_params)
    critic_state = critic_state.replace(params={"sa_encoder": critic_params["sa_encoder"], "g_encoder": critic_params["g_encoder"]})
    
    # wrap it all back into the training_state for easy handling
    training_state = training_state.replace(
        alpha_state=alpha_state,
        actor_state=actor_state,
        critic_state=critic_state,
    )
    
    print(f"Loaded alpha, actor, and critic params from {prev_params_path} and replaced initial parameters in training_state", flush=True)

    # Create directory for figures if it doesn't exist
    # os.makedirs('repeat_reps_figs', exist_ok=True)

    print("Analyzing residuals for all networks...", flush=True)
    actor_residuals, sa_residuals, g_residuals = analyze_all_networks_residuals(
        actor, 
        training_state.actor_state.params,
        sa_encoder,
        training_state.critic_state.params['sa_encoder'],
        g_encoder,
        training_state.critic_state.params['g_encoder'],
        eval_env,
        eval_env_state,
        args
    )

    print("Average residual magnitudes:", flush=True)
    print(f"Actor: {np.mean(actor_residuals):.4f}", flush=True)
    print(f"SA Encoder: {np.mean(sa_residuals):.4f}", flush=True)
    print(f"G Encoder: {np.mean(g_residuals):.4f}", flush=True)

    def deterministic_actor_step(training_state, env, env_state, extra_fields):
        means, _ = actor.apply(training_state.actor_state.params, env_state.obs)
        actions = nn.tanh(means)

        nstate = env.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        
        return nstate, Transition(
            observation=env_state.obs,
            action=actions,
            reward=nstate.reward,
            discount=1-nstate.done,
            extras={"state_extras": state_extras},
        )
   
    # Setting up evaluator
    evaluator = CrlEvaluator(
        deterministic_actor_step,
        eval_env,
        num_eval_envs=args.num_eval_envs,
        episode_length=args.episode_length,
        key=eval_env_key,
    )
    
    # metrics = {}
    # metrics = evaluator.run_evaluation(training_state, metrics)
    # print(f"metrics: {metrics}", flush=True)
    # exit()
    