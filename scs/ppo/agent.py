from __future__ import annotations

from typing import (
    TYPE_CHECKING,
)

from flax import nnx
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

if TYPE_CHECKING:
    from scs.data import (
        TrajectoryData,
        ValueAndGAE,
    )
    from scs.ppo.defaults import PPOConfig
    from scs.ppo.models import ActorCritic


@nnx.jit(static_argnums=(3,))
def actor_action(
    model: ActorCritic,
    states: jax.Array,
    rng: nnx.Rngs,
    config: PPOConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Samples an action from the actor's policy.

    This function computes the action distribution from the actor model, samples
    an action, and exploration noise.
    """
    a_means, a_log_stds, _values = model(states)
    noise = jax.random.normal(rng.noise(), shape=a_means.shape) * config.action_noise
    actions = a_means + jnp.exp(a_log_stds) * jax.random.normal(
        rng.action_select(), shape=a_means.shape
    )
    return actions, noise, a_means, a_log_stds


def loss_fn(
    model: ActorCritic,
    batch: TrajectoryData,
    batch_computations: ValueAndGAE,
    config: PPOConfig,
) -> jax.Array:
    """
    Entropy is based on the differential entropy of a normal distribution:
        H(X)    = log(sigma * sqrt(2 * pi * e))
                = log(sigma) + 0.5 * (log(2 * pi) + 1)
    """
    a_means, a_log_stds, values = model(batch.states)
    values = jnp.squeeze(values)
    returns = batch_computations.gae + batch_computations.values
    value_loss = jnp.mean((returns - values) ** 2).mean()

    entropy = jnp.sum(a_log_stds + 0.5 * (jnp.log(2 * jnp.pi) + 1), axis=-1).mean()

    action_log_densities = norm.logpdf(
        batch.actions, loc=a_means, scale=jnp.exp(a_log_stds)
    )
    density_ratios = jnp.exp(  # Joint density ratio over the "set of actions"
        jnp.sum(action_log_densities - batch.action_log_densities, axis=-1)
    )
    policy_gradient_loss = density_ratios * batch_computations.gae
    clipped_pg_loss = (
        jax.lax.clamp(
            1.0 - config.clip_parameter, density_ratios, 1.0 + config.clip_parameter
        )
        * batch_computations.gae
    )
    ppo_loss = -jnp.mean(jnp.minimum(policy_gradient_loss, clipped_pg_loss), axis=0)
    return (
        ppo_loss
        + config.value_loss_coefficient * value_loss
        + config.entropy_coefficient * entropy
    )
