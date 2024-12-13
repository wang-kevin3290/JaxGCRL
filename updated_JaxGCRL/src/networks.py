from typing import Sequence, Tuple, Callable, NamedTuple, Any

import jax
from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen
import jax.numpy as jnp
from flax.linen.initializers import variance_scaling

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]

@flax.struct.dataclass
class CRLNetworks:
    policy_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution
    sa_encoder: networks.FeedForwardNetwork
    g_encoder: networks.FeedForwardNetwork

#MODIFIED THIS (first equivalently by taking out the last dense layer from the for loop, then I add in skip connections (which if set to 0 should be equivalent))
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
                else:
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


def make_embedder(
    layer_sizes: Sequence[int],
    obs_size: int,
    activation: Callable[[jnp.ndarray], jnp.ndarray] = linen.swish,
    preprocess_observations_fn: types.PreprocessObservationFn = types,
    use_ln: bool = False,
    skip_connections: int = 0,
    clean_jax_arch: bool = False) -> networks.FeedForwardNetwork:

    """Creates a model."""
    dummy_obs = jnp.zeros((1, obs_size))
    if not clean_jax_arch:
        module = MLP(layer_sizes=layer_sizes, activation=activation, use_layer_norm=use_ln, skip_connections=skip_connections)
    else:
        assert all(layer_size == layer_sizes[0] for layer_size in layer_sizes[:-1]), "All layers except the last must be the same size"
        module = MLPCleanJax(network_width=layer_sizes[0], network_depth=len(layer_sizes) - 1, output_size=layer_sizes[-1], skip_connections=skip_connections)

    # TODO: should we have a function to preprocess the observations?
    def apply(processor_params, policy_params, obs):
        # obs = preprocess_observations_fn(obs, processor_params)
        return module.apply(policy_params, obs)

    model = networks.FeedForwardNetwork(init=lambda rng: module.init(rng, dummy_obs), apply=apply)
    return model

def make_policy_network(
    param_size: int,
    obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    skip_connections: int = 0,
    clean_jax_arch: bool = False) -> networks.FeedForwardNetwork:
    """Creates a policy network."""
    if not clean_jax_arch:
        policy_module = MLP(layer_sizes=list(hidden_layer_sizes) + [param_size], activation=activation, kernel_init=jax.nn.initializers.lecun_uniform(), skip_connections=skip_connections)
    else:
        assert all(layer_size == hidden_layer_sizes[0] for layer_size in hidden_layer_sizes[:-1]), "All layers except the last must be the same size"
        policy_module = MLPCleanJax(network_width=hidden_layer_sizes[0], network_depth=len(hidden_layer_sizes) - 1, output_size=param_size, skip_connections=skip_connections)

    def apply(processor_params, policy_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(policy_params, obs)

    dummy_obs = jnp.zeros((1, obs_size))
    return networks.FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_obs), apply=apply)

def make_inference_fn(crl_networks: CRLNetworks):
    """Creates params and inference function for the CRL agent."""
    def make_policy(params: types.PolicyParams, deterministic: bool = False) -> types.Policy:
        def policy(obs: types.Observation, key_sample: PRNGKey) -> Tuple[types.Action, types.Extra]:
            logits = crl_networks.policy_network.apply(*params[:2], obs)
            if deterministic:
                action = crl_networks.parametric_action_distribution.mode(logits)
            else:
                action = crl_networks.parametric_action_distribution.sample(logits, key_sample)
            return action, {}
        return policy
    return make_policy


def make_crl_networks(
    config: NamedTuple,
    env: object,
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    use_ln: bool= False,
    skip_connections: int = 0,
    clean_jax_arch: bool = False
) -> CRLNetworks:
    """Make CRL networks."""
    parametric_action_distribution = distribution.NormalTanhDistribution(event_size=action_size)
    
    policy_network = make_policy_network(
        parametric_action_distribution.param_size,
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        skip_connections=skip_connections,
        clean_jax_arch=clean_jax_arch
    )
    sa_encoder = make_embedder(
        layer_sizes=list(hidden_layer_sizes) + [config.repr_dim],
        obs_size=env.state_dim + action_size,
        activation=activation,
        preprocess_observations_fn=preprocess_observations_fn,
        use_ln=use_ln,
        skip_connections=skip_connections,
        clean_jax_arch=clean_jax_arch
    )
    g_encoder = make_embedder(
        layer_sizes=list(hidden_layer_sizes) + [config.repr_dim],
        obs_size=len(env.goal_indices),
        activation=activation,
        preprocess_observations_fn=preprocess_observations_fn,
        use_ln=use_ln,
        skip_connections=skip_connections,
        clean_jax_arch=clean_jax_arch
    )

    return CRLNetworks(
        policy_network=policy_network,
        parametric_action_distribution=parametric_action_distribution,
        sa_encoder=sa_encoder,
        g_encoder=g_encoder,
    )
