"""TD3 networks."""

from typing import Sequence, Tuple

import jax.numpy as jnp
from brax.training import networks
from brax.training import types
from brax.training.networks import ActivationFn, FeedForwardNetwork, Initializer, MLP
from brax.training.types import PRNGKey
from flax import linen, struct
import jax
from flax.linen.initializers import variance_scaling


@struct.dataclass
class TD3Networks:
    policy_network: networks.FeedForwardNetwork
    q_network: networks.FeedForwardNetwork
    
    
class MLP(linen.Module):
    """MLP module."""

    layer_sizes: Sequence[int]
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True
    use_layer_norm: bool = False
    skip_connections: int = 0

    @linen.compact
    def __call__(self, data: jnp.ndarray):
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes[:-1]):
            hidden = linen.Dense(hidden_size, name=f"hidden_{i}", kernel_init=self.kernel_init, use_bias=self.bias)(hidden)
            if self.use_layer_norm:
                hidden = linen.LayerNorm()(hidden)
            hidden = self.activation(hidden)
            
            if self.skip_connections > 0:
                if i == 0:
                    skip = hidden
                if i > 0 and i % self.skip_connections == 0:
                    hidden = hidden + skip
                    skip = hidden
        
        hidden = linen.Dense(self.layer_sizes[-1], name=f"hidden_{len(self.layer_sizes)-1}", kernel_init=self.kernel_init, use_bias=self.bias)(hidden)
        if self.activate_final:
            if self.use_layer_norm:
                hidden = linen.LayerNorm()(hidden)
            hidden = self.activation(hidden)
        return hidden
    
class MLPCleanJax(linen.Module):
    norm_type = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    output_size: int = 64
    skip_connections: int = 0
    use_relu: int = 0
    @linen.compact
    def __call__(self, data: jnp.ndarray):
        x = data
        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = linen.initializers.zeros

        if self.norm_type == "layer_norm":
            normalize = lambda x: linen.LayerNorm()(x)
        else:
            normalize = lambda x: x
        
        if self.use_relu:
            activation = linen.relu
        else:
            activation = linen.swish
        
        for i in range(self.network_depth):
            x = linen.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
            x = normalize(x)
            x = activation(x)
            
            if self.skip_connections:
                if i == 0:
                    skip = x
                if i > 0 and i % self.skip_connections == 0:
                    x = x + skip
                    skip = x
        
        x = linen.Dense(self.output_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x


def make_inference_fn(td3_networks: TD3Networks):
    """Creates params and inference function for the TD3 agent."""

    def make_policy(params: types.PolicyParams, exploration_noise=0.0, noise_clip=0.0, deterministic=False) -> types.Policy:
        def policy(observations: types.Observation,
                   key_noise: PRNGKey) -> Tuple[types.Action, types.Extra]:
            actions = td3_networks.policy_network.apply(*params, observations)
            noise = (jax.random.normal(key_noise, actions.shape) * exploration_noise).clip(-noise_clip, noise_clip)
            return actions + noise, {}

        return policy

    return make_policy


def make_policy_network(
        param_size: int,
        obs_size: int,
        preprocess_observations_fn: types.PreprocessObservationFn = types
        .identity_observation_preprocessor,
        hidden_layer_sizes: Sequence[int] = (256, 256),
        activation: ActivationFn = linen.relu,
        kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
        layer_norm: bool = False,
        skip_connections: int = 4,
        clean_jax_arch: bool = True) -> FeedForwardNetwork:
    """Creates a policy network."""
    
    if not clean_jax_arch:
        policy_module = MLP(
            layer_sizes=list(hidden_layer_sizes) + [param_size],
            activation=activation,
            kernel_init=kernel_init,
            layer_norm=layer_norm)
    else:
        policy_module = MLPCleanJax(
            network_width=hidden_layer_sizes[0],
            network_depth=len(hidden_layer_sizes) - 1,
            output_size=param_size,
            skip_connections=skip_connections)

    def apply(processor_params, policy_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        raw_actions = policy_module.apply(policy_params, obs)
        return linen.tanh(raw_actions)

    dummy_obs = jnp.zeros((1, obs_size))
    return FeedForwardNetwork(init=lambda key: policy_module.init(key, dummy_obs), apply=apply)

def make_q_network(
    obs_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    n_critics: int = 2,
    skip_connections: int = 4,
    clean_jax_arch: bool = False) -> networks.FeedForwardNetwork:
  """Creates a value network."""

  class QModule(linen.Module):
    """Q Module."""
    n_critics: int

    @linen.compact
    def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
      hidden = jnp.concatenate([obs, actions], axis=-1)
      res = []
      for _ in range(self.n_critics):
        if not clean_jax_arch:
            q = MLP(layer_sizes=list(hidden_layer_sizes) + [1], activation=activation, kernel_init=jax.nn.initializers.lecun_uniform(), skip_connections=skip_connections)(hidden)
        else:
            assert all(layer_size == hidden_layer_sizes[0] for layer_size in hidden_layer_sizes[:-1]), "All layers except the last must be the same size"
            q = MLPCleanJax(network_width=hidden_layer_sizes[0], network_depth=len(hidden_layer_sizes) - 1, output_size=1, skip_connections=skip_connections)(hidden)
        res.append(q)
      return jnp.concatenate(res, axis=-1)

  q_module = QModule(n_critics=n_critics)

  def apply(processor_params, q_params, obs, actions):
    obs = preprocess_observations_fn(obs, processor_params)
    return q_module.apply(q_params, obs, actions)

  dummy_obs = jnp.zeros((1, obs_size))
  dummy_action = jnp.zeros((1, action_size))
  return networks.FeedForwardNetwork(
      init=lambda key: q_module.init(key, dummy_obs, dummy_action), apply=apply)


def make_td3_networks(
        observation_size: int,
        action_size: int,
        preprocess_observations_fn: types.PreprocessObservationFn = types
        .identity_observation_preprocessor,
        hidden_layer_sizes: Sequence[int] = (256, 256),
        activation: networks.ActivationFn = linen.relu,
        skip_connections: int = 4,
        clean_jax_arch: bool = True) -> TD3Networks:
    """Make TD3 networks."""
    policy_network = make_policy_network(
        action_size,
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        skip_connections=skip_connections,
        clean_jax_arch=clean_jax_arch
    )

    q_network = make_q_network(
        observation_size,
        action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation)

    return TD3Networks(
        policy_network=policy_network,
        q_network=q_network)


