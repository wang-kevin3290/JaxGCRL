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

from brax import envs
from etils import epath
from dataclasses import dataclass
from collections import namedtuple
from typing import NamedTuple, Any
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
        
        print(f"x.shape: {x.shape}", flush=True)

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
        
def gpu_warmup():
    """
    Dummy code to perform some GPU utilization at the beginning
    so the cluster doesn't kill the job for inactivity.
    """
    print("Starting GPU warmup...", flush=True)
    import jax
    import jax.numpy as jnp

    # A quick matrix multiplication loop that exerts GPU usage
    x = jnp.ones((1024, 1024))
    y = jnp.ones((1024, 1024))
    for _ in range(20):
        x = jnp.dot(x, y)
    x.block_until_ready()
    print("GPU warmup complete.", flush=True)
    
if __name__ == "__main__":   

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, default="/scratch/gpfs/kw6487/JaxGCRL/clean_JaxGCRL/runs/humanoid_271_20250117-071637",
                        help='Path to previous run folder containing args.pkl and final.pkl')
    parser.add_argument('--repeat2', type=str, default="",
                        help='Number of repeat layers for actor 2 (comma separated, so you can do "0,4,8,12"); should be multiple of args.actor_skip_connections')
    parser.add_argument('--repeat3', type=str, default="",
                        help='Number of repeat layers for actor 3 (comma separated, so you can do "0,4,8,12"); should be multiple of args.actor_skip_connections')
    cmd_args = parser.parse_args()

    prev_run_folder = cmd_args.run
    repeat2 = cmd_args.repeat2
    repeat3 = cmd_args.repeat3
    NUM_REPEAT_LAYERS_2s = [int(x) for x in repeat2.split(',')] if repeat2 else []
    NUM_REPEAT_LAYERS_3s = [int(x) for x in repeat3.split(',')] if repeat3 else []


    # prev_run_folder = "/scratch/gpfs/kw6487/JaxGCRL/clean_JaxGCRL/runs/humanoid_271_20250117-071637"
    print(f"prev_run_folder: {prev_run_folder}", flush=True)
    
    prev_args_path = Path(prev_run_folder) / "args.pkl"
    prev_params_path = Path(prev_run_folder) / "final.pkl"
    # prev_replay_buffer_path = Path(prev_run_folder) / "final_buffer.pkl"     
    # 
    import pickle
    with open(prev_args_path, 'rb') as f:
        args = pickle.load(f)   

    # print(f"args: {args}", flush=True)
        
    
    # Print every arg
    PRINT_ARGS = 0
    if PRINT_ARGS:
        print("Arguments:", flush=True)
        for arg, value in vars(args).items():
            print(f"{arg}: {value}", flush=True)
        print("\n", flush=True)



    key = jax.random.PRNGKey(args.seed)
    key, buffer_key, env_key, eval_env_key, actor_key, sa_key, g_key, sym_key, asym_key, memory_bank_key, actor2_key, actor3_key = jax.random.split(key, 12)

    def setup_actor2(actor_params, NUM_REPEAT_LAYERS): #returns actor2_params
        NUM_REPEAT_RESIDUAL_BLOCKS = NUM_REPEAT_LAYERS // args.actor_skip_connections + 1 #this means a total of 2, not 2 repeat (i.e. 1 means no repeats, recover original)
        NEW_DEPTH = args.actor_depth + (NUM_REPEAT_RESIDUAL_BLOCKS - 1) * args.actor_skip_connections
        print(f"NUM_REPEAT_RESIDUAL_BLOCKS: {NUM_REPEAT_RESIDUAL_BLOCKS}")
        print(f"OLD_DEPTH: {args.actor_depth}, NEW_DEPTH: {NEW_DEPTH}")

        actor2 = Actor(action_size=action_size, network_width=args.actor_network_width, network_depth=NEW_DEPTH, skip_connections=args.actor_skip_connections, use_relu=args.use_relu)
        # key, actor2_key = jax.random.split(key, 2)
        actor2_params = actor2.init(actor2_key, np.ones([1, obs_size]))

        dense_mappings = {} #maps actor2_params idx -> actor_params idx
        N_actor, N_actor2 = args.actor_depth + 3, NEW_DEPTH + 3 #the num of Dense layers for each i.e 11, 15 (in our example case)
        assert len([x for x in actor_params['params'].keys() if 'Dense' in x]) == N_actor
        assert len([x for x in actor2_params['params'].keys() if 'Dense' in x]) == N_actor2
        print(N_actor, N_actor2)

        copy_range = np.arange(N_actor-2-args.actor_skip_connections, N_actor-2) #INCLUSIVE, the range of the layers in actor that are being copied

        #okay so let's just copy over everything to start (based on mod), then we'll fix
        for i in range(N_actor2):
            idx = (i-1) % 4
            dense_mappings[i] = copy_range[idx]

        #now we fix
        dense_mappings[0] = 0
        for i in range(1, args.actor_depth+1):
            dense_mappings[i] = i
        dense_mappings[N_actor2-2] = N_actor-2
        dense_mappings[N_actor2-1] = N_actor-1

        # for k, v in dense_mappings.items():
        #     print(f"Dense_{k} <- Dense_{v}")

        for k, v in dense_mappings.items():
            str_k, str_v = f'Dense_{k}', f'Dense_{v}'
            actor2_params['params'][str_k] = actor_params['params'][str_v]
            print(f"Setting actor2_params's {str_k} with actor_params's {str_v}")

            ln_mappings = {} #maps actor2_params idx -> actor_params idx
        LN_actor, LN_actor2 = args.actor_depth + 1, NEW_DEPTH + 1 
        assert len([x for x in actor_params['params'].keys() if 'LayerNorm' in x]) == LN_actor
        assert len([x for x in actor2_params['params'].keys() if 'LayerNorm' in x]) == LN_actor2
        print(LN_actor, LN_actor2)

        copy_range = np.arange(LN_actor-args.actor_skip_connections, LN_actor) #should be equivalent


        #okay so let's just copy over everything to start (based on mod), then we'll fix
        for i in range(LN_actor2):
            idx = (i-1) % 4
            ln_mappings[i] = copy_range[idx]

        #now we fix
        ln_mappings[0] = 0
        for i in range(1, args.actor_depth+1):
            ln_mappings[i] = i

        # for k, v in ln_mappings.items():
        #     print(f"Dense_{k} <- Dense_{v}")

        for k, v in ln_mappings.items():
            str_k, str_v = f'LayerNorm_{k}', f'LayerNorm_{v}'
            actor2_params['params'][str_k] = actor_params['params'][str_v]
            print(f"Setting actor2_params's {str_k} with actor_params's {str_v}")
        
        return actor2, actor2_params

    def setup_actor3(actor_params, NUM_REPEAT_LAYERS): #returns actor3_params
        ### NOW WE WORK ON ACTOR 3
        NEW_DEPTH = args.actor_depth + NUM_REPEAT_LAYERS
        print(f"NUM_REPEAT_LAYERS: {NUM_REPEAT_LAYERS}")
        print(f"OLD_DEPTH: {args.actor_depth}, NEW_DEPTH: {NEW_DEPTH}")

        actor3 = Actor(action_size=action_size, network_width=args.actor_network_width, network_depth=NEW_DEPTH, skip_connections=args.actor_skip_connections, use_relu=args.use_relu)
        # key, actor3_key = jax.random.split(key, 2)
        actor3_params = actor3.init(actor3_key, np.ones([1, obs_size]))

        dense_mappings = {} #maps actor3_params idx -> actor_params idx
        N_actor, N_actor3 = args.actor_depth + 3, NEW_DEPTH + 3 #the num of Dense layers for each i.e 11, 15 (in our example case)
        assert len([x for x in actor_params['params'].keys() if 'Dense' in x]) == N_actor
        assert len([x for x in actor3_params['params'].keys() if 'Dense' in x]) == N_actor3
        print(N_actor, N_actor3)

        # copy_range = np.arange(N_actor-2-args.actor_skip_connections, N_actor-2) #INCLUSIVE, the range of the layers in actor that are being copied
        copy_idx = args.actor_depth

        #okay so let's just copy over everything to start (based on mod), then we'll fix
        for i in range(N_actor3):
            dense_mappings[i] = copy_idx

        #now we fix
        dense_mappings[0] = 0
        for i in range(1, args.actor_depth+1):
            dense_mappings[i] = i
        dense_mappings[N_actor3-2] = N_actor-2
        dense_mappings[N_actor3-1] = N_actor-1

        # for k, v in dense_mappings.items():
        #     print(f"Dense_{k} <- Dense_{v}")

        for k, v in dense_mappings.items():
            str_k, str_v = f'Dense_{k}', f'Dense_{v}'
            actor3_params['params'][str_k] = actor_params['params'][str_v]
            print(f"Setting actor3_params's {str_k} with actor_params's {str_v}")

        ln_mappings = {} #maps actor3_params idx -> actor_params idx
        LN_actor, LN_actor3 = args.actor_depth + 1, NEW_DEPTH + 1 
        assert len([x for x in actor_params['params'].keys() if 'LayerNorm' in x]) == LN_actor
        assert len([x for x in actor3_params['params'].keys() if 'LayerNorm' in x]) == LN_actor3
        print(LN_actor, LN_actor3)

        # copy_range = np.arange(LN_actor-args.actor_skip_connections, LN_actor) #should be equivalent
        copy_idx = args.actor_depth

        #okay so let's just copy over everything to start (based on mod), then we'll fix
        for i in range(LN_actor3):
            ln_mappings[i] = copy_idx

        #now we fix
        ln_mappings[0] = 0
        for i in range(1, args.actor_depth+1):
            ln_mappings[i] = i

        # for k, v in ln_mappings.items():
        #     print(f"Dense_{k} <- Dense_{v}")

        for k, v in ln_mappings.items():
            str_k, str_v = f'LayerNorm_{k}', f'LayerNorm_{v}'
            actor3_params['params'][str_k] = actor_params['params'][str_v]
            print(f"Setting actor3_params's {str_k} with actor_params's {str_v}")

        return actor3, actor3_params


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
        
    
    # Trainstate
    training_state = TrainingState(
        env_steps=jnp.zeros(()),
        gradient_steps=jnp.zeros(()),
        actor_state=actor_state,
        critic_state=None,
        alpha_state=None,
        memory_bank_state=None,
    )
    
    # If continuing from a previous run, load the saved parameters and OVERWRITE the initial parameters
    # if args.load_prev_ckpt:        
    from brax.io import model
    try:
        params = model.load_params(prev_params_path)
        alpha_params, actor_params, critic_params = params
        sa_encoder_params, g_encoder_params = critic_params['sa_encoder'], critic_params['g_encoder']
        print(f"Loaded alpha, actor, and critic params from {prev_params_path}", flush=True)
    except:
        print(f"Failed to load params from {prev_params_path}", flush=True)
        
        
    # replace the initial parameters with the loaded ones
    # alpha_state = alpha_state.replace(params=alpha_params)
    actor_state = actor_state.replace(params=actor_params)
    # critic_state = critic_state.replace(params={"sa_encoder": sa_encoder_params, "g_encoder": g_encoder_params})
    
    # wrap it all back into the training_state for easy handling
    training_state = training_state.replace(
        # alpha_state=alpha_state,
        actor_state=actor_state,
        # critic_state=critic_state,
    )
    
    print(f"Loaded alpha, actor, and critic params from {prev_params_path} and replaced initial parameters in training_state", flush=True)


    ### SET UP ACTOR 2
    # NUM_REPEAT_LAYERS_2 = args.actor_skip_connections * 1 * 0 #this is added layers
    # NUM_REPEAT_LAYERS_3 = args.actor_skip_connections * 1 * 0 #this is added layers

    def run_actor2(NUM_REPEAT_LAYERS): #runs an experiment with actor2 setup given NUM_REPEAT_LAYERS layers to repeat
        actor2, actor2_params = setup_actor2(actor_params, NUM_REPEAT_LAYERS)
        actor2_state = TrainState.create(
            apply_fn=actor2.apply,
            params=actor2_params,
            tx=optax.adam(learning_rate=args.actor_lr)
        )
        training_state2 = TrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            actor_state=actor2_state,
            critic_state=None,
            alpha_state=None,
            memory_bank_state=None,
        )

        def deterministic_actor2_step(training_state, env, env_state, extra_fields):
            means, _ = actor2.apply(training_state.actor_state.params, env_state.obs)
            actions = nn.tanh( means )

            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}
            
            return nstate, Transition(
                observation=env_state.obs,
                action=actions,
                reward=nstate.reward,
                discount=1-nstate.done,
                extras={"state_extras": state_extras},
            )
        evaluator2 = CrlEvaluator(
            deterministic_actor2_step,
            eval_env,
            num_eval_envs=args.num_eval_envs,
            episode_length=args.episode_length,
            key=eval_env_key,
        )

        metrics2 = {}
        metrics2 = evaluator2.run_evaluation(training_state2, metrics2)

        return metrics2
    
    def run_actor3(NUM_REPEAT_LAYERS): #runs an experiment with actor2 setup given NUM_REPEAT_LAYERS layers to repeat
        actor3, actor3_params = setup_actor3(actor_params, NUM_REPEAT_LAYERS)
        actor3_state = TrainState.create(
            apply_fn=actor3.apply,
            params=actor3_params,
            tx=optax.adam(learning_rate=args.actor_lr)
        )
        training_state3 = TrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            actor_state=actor3_state,
            critic_state=None,
            alpha_state=None,
            memory_bank_state=None,
        )
        def deterministic_actor3_step(training_state, env, env_state, extra_fields):
            means, _ = actor3.apply(training_state.actor_state.params, env_state.obs)
            actions = nn.tanh( means )

            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}
            
            return nstate, Transition(
                observation=env_state.obs,
                action=actions,
                reward=nstate.reward,
                discount=1-nstate.done,
                extras={"state_extras": state_extras},
            )
        evaluator3 = CrlEvaluator(
            deterministic_actor3_step,
            eval_env,
            num_eval_envs=args.num_eval_envs,
            episode_length=args.episode_length,
            key=eval_env_key,
        )
        print('\n')
        metrics3 = {}
        metrics3 = evaluator3.run_evaluation(training_state3, metrics3)
        # print(f"metrics3: {metrics3}", flush=True)
        # print(f"metrics3['eval/episode_success']: {metrics3['eval/episode_success']}", flush=True)
        # print(f"metrics3['eval/episode_success_any']: {metrics3['eval/episode_success_any']}", flush=True)
        return metrics3
  


    def deterministic_actor_step(training_state, env, env_state, extra_fields):
        means, _ = actor.apply(training_state.actor_state.params, env_state.obs)
        actions = nn.tanh( means )

        nstate = env.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        
        return nstate, Transition(
            observation=env_state.obs,
            action=actions,
            reward=nstate.reward,
            discount=1-nstate.done,
            extras={"state_extras": state_extras},
        )
    
   
    '''Setting up evaluator'''
    evaluator = CrlEvaluator(
        deterministic_actor_step,
        eval_env,
        num_eval_envs=args.num_eval_envs,
        episode_length=args.episode_length,
        key=eval_env_key,
    )

    metrics2s = []
    metrics3s = []
    for NUM_REPEAT_LAYERS_2 in NUM_REPEAT_LAYERS_2s:
        metrics2 = run_actor2(NUM_REPEAT_LAYERS_2)
        metrics2s.append(metrics2)
    for NUM_REPEAT_LAYERS_3 in NUM_REPEAT_LAYERS_3s:
        metrics3 = run_actor3(NUM_REPEAT_LAYERS_3)
        metrics3s.append(metrics3)

    metrics = {}
    metrics = evaluator.run_evaluation(training_state, metrics)
    print(f" ---- ORIGINAL ----")
    print(f"metrics: {metrics}", flush=True)
    print(f"metrics['eval/episode_success']: {metrics['eval/episode_success']}", flush=True)
    print(f"metrics['eval/episode_success_any']: {metrics['eval/episode_success_any']}", flush=True)

    print('\n')

    for i, metrics2 in enumerate(metrics2s):
        print(f" ---- (ACTOR 2, REPEAT RESIDUAL BLOCK) NUM REPEATED LAYERS: {NUM_REPEAT_LAYERS_2s[i]} ----")
        print(f"metrics2: {metrics2}", flush=True)
        print(f"metrics2['eval/episode_success']: {metrics2['eval/episode_success']}", flush=True)
        print(f"metrics2['eval/episode_success_any']: {metrics2['eval/episode_success_any']}", flush=True)
        print('\n')

    print('\n')

    for i, metrics3 in enumerate(metrics3s):
        print(f" ---- (ACTOR 3, REPEAT ONLY LAST LAYER) NUM REPEATED LAYERS: {NUM_REPEAT_LAYERS_3s[i]} ----")
        print(f"metrics3: {metrics3}", flush=True)
        print(f"metrics3['eval/episode_success']: {metrics3['eval/episode_success']}", flush=True)
        print(f"metrics3['eval/episode_success_any']: {metrics3['eval/episode_success_any']}", flush=True)
        print('\n')



    
    import matplotlib.pyplot as plt
    import numpy as np

    # Set up data for plotting
    labels_orig = ['Original']
    labels2 = [f'Repeat {num_layers} layers (depth {args.actor_depth + num_layers})' for num_layers in NUM_REPEAT_LAYERS_2s]
    labels3 = [f'Repeat {num_layers} layers (depth {args.actor_depth + num_layers})' for num_layers in NUM_REPEAT_LAYERS_3s]
    
    success_rate_orig = [metrics['eval/episode_success']]
    success_rates2 = [metrics2['eval/episode_success'] for metrics2 in metrics2s]
    success_rates3 = [metrics3['eval/episode_success'] for metrics3 in metrics3s]

    plt.rcParams.update({'font.size': 14})  # Increase base font size

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,10))
    width = 0.8

    # First subplot - Original vs Actor2
    x1 = np.arange(len(labels_orig + labels2))
    rects1 = ax1.bar(x1, success_rate_orig + success_rates2, width, color='royalblue')
    ax1.set_ylabel(f'({args.env_id}) Success', fontsize=16)
    ax1.set_title(f'REPEAT LAST RESIDUAL BLOCK: Original (Depth {args.actor_depth}) vs Repeated: \n(Depths {", ".join(str(args.actor_depth + x) for x in NUM_REPEAT_LAYERS_2s)})', 
                  fontsize=18, pad=20)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(labels_orig + labels2, fontsize=10)
    ax1.tick_params(axis='y', labelsize=14)

    # Second subplot - Original vs Actor3  
    x2 = np.arange(len(labels_orig + labels3))
    rects2 = ax2.bar(x2, success_rate_orig + success_rates3, width, color='darkgreen')
    ax2.set_ylabel(f'({args.env_id}) Success', fontsize=16)
    ax2.set_title(f'REPEAT ONLY LAST LAYER: Original (Depth {args.actor_depth}) vs Repeated: \n(Depths {", ".join(str(args.actor_depth + x) for x in NUM_REPEAT_LAYERS_3s)})',
                  fontsize=18, pad=20)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels_orig + labels3, fontsize=10)
    ax2.tick_params(axis='y', labelsize=14)

    # Add value labels on top of bars
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=14)

    autolabel(rects1, ax1)
    autolabel(rects2, ax2)

    plt.tight_layout()
    plt.savefig(f'repeat_reps_figs/{args.env_id}_{args.seed}_success_rates_comparison_{args.actor_depth}_{repeat2}_{repeat3}.png')
    print(f"Saved plot: repeat_reps_figs/{args.env_id}_{args.seed}_success_rates_comparison_{args.actor_depth}_{repeat2}_{repeat3}.png", flush=True)
    plt.close()

    exit()
    