# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PPO networks."""

from typing import Sequence, Tuple, Callable, Any

from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen
import jax
from flax.linen.initializers import variance_scaling
import jax.numpy as jnp

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


@flax.struct.dataclass
class PPONetworks:
  policy_network: networks.FeedForwardNetwork
  value_network: networks.FeedForwardNetwork
  parametric_action_distribution: distribution.ParametricDistribution
  
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


def make_inference_fn(ppo_networks: PPONetworks):
  """Creates params and inference function for the PPO agent."""

  def make_policy(params: types.PolicyParams,
                  deterministic: bool = False) -> types.Policy:
    policy_network = ppo_networks.policy_network
    parametric_action_distribution = ppo_networks.parametric_action_distribution

    def policy(observations: types.Observation,
               key_sample: PRNGKey) -> Tuple[types.Action, types.Extra]:
      logits = policy_network.apply(*params, observations)
      if deterministic:
        return ppo_networks.parametric_action_distribution.mode(logits), {}
      raw_actions = parametric_action_distribution.sample_no_postprocessing(
          logits, key_sample)
      log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
      postprocessed_actions = parametric_action_distribution.postprocess(
          raw_actions)
      return postprocessed_actions, {
          'log_prob': log_prob,
          'raw_action': raw_actions
      }

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
    clean_jax_arch: bool = True) -> networks.FeedForwardNetwork:
  """Creates a policy network."""
  if not clean_jax_arch:
    policy_module = MLP(layer_sizes=list(hidden_layer_sizes) + [param_size], activation=activation, kernel_init=kernel_init, use_layer_norm=layer_norm, skip_connections=skip_connections)
  else:
    assert all(layer_size == hidden_layer_sizes[0] for layer_size in hidden_layer_sizes[:-1]), "All layers except the last must be the same size"
    policy_module = MLPCleanJax(network_width=hidden_layer_sizes[0], network_depth=len(hidden_layer_sizes) - 1, output_size=param_size, skip_connections=skip_connections)

  def apply(processor_params, policy_params, obs):
    obs = preprocess_observations_fn(obs, processor_params)
    return policy_module.apply(policy_params, obs)

  dummy_obs = jnp.zeros((1, obs_size))
  return networks.FeedForwardNetwork(
      init=lambda key: policy_module.init(key, dummy_obs), apply=apply)

def make_value_network(
    obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    skip_connections: int = 4,
    clean_jax_arch: bool = True) -> networks.FeedForwardNetwork:
  """Creates a policy network."""
  if not clean_jax_arch:
    value_module = MLP(
        layer_sizes=list(hidden_layer_sizes) + [1],
        activation=activation,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        skip_connections=skip_connections)
  else:
    assert all(layer_size == hidden_layer_sizes[0] for layer_size in hidden_layer_sizes[:-1]), "All layers except the last must be the same size"
    value_module = MLPCleanJax(network_width=hidden_layer_sizes[0], network_depth=len(hidden_layer_sizes) - 1, output_size=1, skip_connections=skip_connections)

  def apply(processor_params, policy_params, obs):
    obs = preprocess_observations_fn(obs, processor_params)
    return jnp.squeeze(value_module.apply(policy_params, obs), axis=-1)

  dummy_obs = jnp.zeros((1, obs_size))
  return networks.FeedForwardNetwork(
      init=lambda key: value_module.init(key, dummy_obs), apply=apply)

def make_ppo_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    activation: networks.ActivationFn = linen.swish,
    skip_connections: int = 4,
    clean_jax_arch: bool = True) -> PPONetworks:
  """Make PPO networks with preprocessor."""
  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=action_size)
  policy_network = make_policy_network(
      parametric_action_distribution.param_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=policy_hidden_layer_sizes,
      activation=activation,
      skip_connections=skip_connections,
      clean_jax_arch=clean_jax_arch)
  value_network = make_value_network(
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=value_hidden_layer_sizes,
      activation=activation,
      skip_connections=skip_connections,
      clean_jax_arch=clean_jax_arch)

  return PPONetworks(
      policy_network=policy_network,
      value_network=value_network,
      parametric_action_distribution=parametric_action_distribution)
