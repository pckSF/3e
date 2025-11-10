from __future__ import annotations

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
    get_advantage_batch,
    get_trajectory_batch,
)

if TYPE_CHECKING:
    from scs.nn_modules import NNTrainingState
    from scs.ppo.defaults import PPOConfig
    from scs.ppo.models import PolicyValue


@nnx.jit
def actor_action(
    model: PolicyValue,
    states: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Samples an action from the actor's policy.

    This function computes the action distribution from the actor model and
    samples an action.
    """
    a_means, a_log_stds, _values = model(states)
    actions = a_means + jnp.exp(a_log_stds) * jax.random.normal(
        key, shape=a_means.shape
    )
    return actions, a_means, a_log_stds


def loss_fn(
    model: PolicyValue,
    batch: TrajectoryData,
    batch_computations: TrajectoryGAE,
    config: PPOConfig,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    """Computes the PPO loss for a batch of trajectory data.

    This function calculates the combined loss for the actor-critic model, which
    includes the policy loss (PPO's clipped objective), the value function loss,
    and an entropy bonus to encourage exploration.

    The entropy is based on the differential entropy of a normal distribution:
        H(X)    = log(sigma * sqrt(2 * pi * e))
                = log(sigma) + 0.5 * (log(2 * pi) + 1)

    Since the optax optimizers perform gradient descent, we return the negative
    of the total loss. Reminder, the loss is composed of a
    -   ppo_value, which are the weighted advantages that we want to maximize.
    -   value_loss, which we want to minimize.
    -   entropy, which we want to maximize.

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
    value_loss = jnp.mean((returns - values) ** 2)
    entropy = jnp.sum(a_log_stds + 0.5 * (jnp.log(2 * jnp.pi) + 1), axis=-1).mean()

    action_log_densities = norm.logpdf(
        batch.actions, loc=a_means, scale=jnp.exp(a_log_stds)
    )
    # Density ratio for the action vectors that constitute an action
    log_ratios = jnp.sum(action_log_densities - batch.action_log_densities, axis=-1)
    density_ratios = jnp.exp(log_ratios)
    kl_estimate = ((density_ratios - 1) - log_ratios).mean()

    policy_gradient_value = density_ratios * batch_computations.advantages
    clipped_pg_value = (
        jax.lax.clamp(
            1.0 - config.clip_parameter, density_ratios, 1.0 + config.clip_parameter
        )
        * batch_computations.advantages
    )
    ppo_value = jnp.mean(jnp.minimum(policy_gradient_value, clipped_pg_value), axis=0)
    return -(
        ppo_value
        - config.value_loss_coefficient * value_loss
        + config.entropy_coefficient * entropy
    ), (ppo_value, value_loss, entropy, kl_estimate)


def train_step(
    train_state: NNTrainingState,
    batch_indices: jax.Array,
    trajectory: TrajectoryData,
    trajectory_advantages: TrajectoryGAE,
    config: PPOConfig,
) -> tuple[
    NNTrainingState, tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]
]:
    """Performs a single training step on a batch of data.

    This function is designed to be used with `jax.lax.scan` to iterate over
    a set of batch indices.

    Args:
        train_state: The current training state, acting as the carry in a scan.
        batch_indices: The indices for the data batch to be processed.
        trajectories: The full stacked trajectory of all agent data for the epoch.
        trajectory_advantages: The pre-computed advantages for the full trajectory.
        config: The agent's configuration.

    Returns:
        A tuple containing the updated training state and the loss for the batch.
    """
    model = nnx.merge(train_state.model_def, train_state.model_state)
    batch = get_trajectory_batch(trajectory, batch_indices)
    batch_computations = get_advantage_batch(trajectory_advantages, batch_indices)
    grad_fn = nnx.value_and_grad(loss_fn, argnums=0, has_aux=True)
    (loss, *loss_components), grads = grad_fn(model, batch, batch_computations, config)
    return train_state.apply_gradients(grads), (loss, loss_components)
