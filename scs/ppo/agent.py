from __future__ import annotations

from typing import (
    TYPE_CHECKING,
)

from flax import nnx
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from scs.data import TrajectoryData
    from scs.ppo.defaults import PPOConfig
    from scs.ppo.models import ActorCritic


@nnx.jit(static_argnums=(3,))
def actor_action(
    model: ActorCritic,
    states: jax.Array,
    rng: nnx.Rngs,
    config: PPOConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Samples an action from the actor's policy.

    This function computes the action distribution from the actor model, samples
    an action, and exploration noise.
    """
    a_means, a_log_stds, values = model(states)
    noise = jax.random.normal(rng.noise(), shape=a_means.shape) * config.action_noise
    actions = a_means + jnp.exp(a_log_stds) * jax.random.normal(
        rng.action_select(), shape=a_means.shape
    )
    return actions, noise, a_means, a_log_stds, values


def loss_fn(
    model: ActorCritic,
    batch: TrajectoryData,
    config: PPOConfig,
) -> jax.Array:
    a_means, a_log_stds, values = model(batch.states)
    a_means, a_log_stds, values = a_means[:, 0], a_log_stds[:, 0], values[:, 0]
