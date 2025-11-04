from __future__ import annotations

from functools import partial
from typing import (
    TYPE_CHECKING,
)

from flax import nnx
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from scs.data import (
    TrajectoryData,
    TrajectoryGAE,
    compute_advantages,
    get_advantage_batch,
    get_trajectory_batch,
)

if TYPE_CHECKING:
    from scs.nn_modules import NNTrainingState
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
    batch_computations: TrajectoryGAE,
    config: PPOConfig,
) -> jax.Array:
    """Computes the PPO loss for a batch of trajectory data.

    This function calculates the combined loss for the actor-critic model, which
    includes the policy loss (PPO's clipped objective), the value function loss,
    and an entropy bonus to encourage exploration.

    The entropy is based on the differential entropy of a normal distribution:
        H(X)    = log(sigma * sqrt(2 * pi * e))
                = log(sigma) + 0.5 * (log(2 * pi) + 1)

    Args:
        model: The actor-critic model being trained.
        batch: A batch of trajectory data from rollouts.
        batch_computations: Pre-computed values like GAE and returns.
        config: The agent's configuration.

    Returns:
        The total PPO loss for the batch.
    """
    a_means, a_log_stds, values = model(batch.states)
    values = jnp.squeeze(values)
    returns = batch_computations.advantages + batch_computations.values
    value_loss = jnp.mean((returns - values) ** 2).mean()

    entropy = jnp.sum(a_log_stds + 0.5 * (jnp.log(2 * jnp.pi) + 1), axis=-1).mean()

    action_log_densities = norm.logpdf(
        batch.actions, loc=a_means, scale=jnp.exp(a_log_stds)
    )
    density_ratios = jnp.exp(  # Density ratio for the action vectors
        jnp.sum(action_log_densities - batch.action_log_densities, axis=-1)
    )
    policy_gradient_loss = density_ratios * batch_computations.advantages
    clipped_pg_loss = (
        jax.lax.clamp(
            1.0 - config.clip_parameter, density_ratios, 1.0 + config.clip_parameter
        )
        * batch_computations.advantages
    )
    ppo_loss = -jnp.mean(jnp.minimum(policy_gradient_loss, clipped_pg_loss), axis=0)
    return (
        ppo_loss
        + config.value_loss_coefficient * value_loss
        + config.entropy_coefficient * entropy
    )


@partial(jax.jit, static_argnames=("config",))
def train_step(
    train_state: NNTrainingState,
    batch_indices: jax.Array,
    trajectory: TrajectoryData,
    config: PPOConfig,
) -> tuple[NNTrainingState, jax.Array]:
    """Performs a single training step on a batch of data.

    This function is designed to be used with `jax.lax.scan` to iterate over
    a set of batch indices.

    Args:
        train_state: The current training state, acting as the carry in a scan.
        batch_indices: The indices for the data batch to be processed.
        trajectory: The full trajectory data for the epoch.
        config: The agent's configuration.

    Returns:
        A tuple containing the updated training state and the loss for the batch.
    """
    model = nnx.merge(train_state.model_def, train_state.model_state)
    trajectory_computations = compute_advantages(
        trajectory=trajectory,
        model=model,
        config=config,
    )
    batch = get_trajectory_batch(trajectory, batch_indices)
    batch_computations = get_advantage_batch(trajectory_computations, batch_indices)
    grad_fn = nnx.value_and_grad(loss_fn, argnums=0)
    loss, grads = grad_fn(model, batch, batch_computations, config)
    return train_state.apply_gradients(grads), loss
