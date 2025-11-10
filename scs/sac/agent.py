from __future__ import annotations

from typing import (
    TYPE_CHECKING,
)

from flax import nnx
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

if TYPE_CHECKING:
    from scs.data import TrajectoryData
    from scs.nn_modules import (
        NNTrainingState,
        NNTrainingStateSoftTarget,
    )
    from scs.sac.defaults import SACConfig
    from scs.sac.models import (
        Policy,
        QValue,
    )


@nnx.jit
def actor_action(
    model_policy: Policy,
    states: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Samples an action from the actor's policy.

    This function computes the action distribution from the actor model and
    samples an action.
    """
    a_means, a_log_stds = model_policy(states)
    actions = a_means + jnp.exp(a_log_stds) * jax.random.normal(
        key, shape=a_means.shape
    )
    return actions, a_means, a_log_stds


def compute_q_target(
    model_policy: Policy,
    model_q1_target: QValue,
    model_q2_target: QValue,
    batch: TrajectoryData,
    config: SACConfig,
    key: jax.Array,
) -> jax.Array:
    actions, a_means, a_log_stds = actor_action(
        model_policy,
        batch.next_states,
        key,
    )
    action_log_densities = norm.logpdf(
        actions, loc=a_means, scale=jnp.exp(a_log_stds)
    ).sum(axis=-1)
    q1_values = model_q1_target(batch.next_states, actions)
    q2_values = model_q2_target(batch.next_states, actions)
    min_q_values = jnp.minimum(q1_values, q2_values)
    next_values = min_q_values - config.entropy_coefficient * action_log_densities
    next_values *= 1.0 - batch.terminals
    q_targets = batch.rewards + config.discount_factor * next_values
    return q_targets


def qvalue_loss_fn(
    model_qvalue: QValue,
    batch: TrajectoryData,
    target_qvalue: jax.Array,
) -> jax.Array:
    q_values = model_qvalue(batch.states, batch.actions)
    return jnp.mean((target_qvalue - q_values) ** 2)


def policy_loss_fn(
    model_policy: Policy,
    model_q1: QValue,
    model_q2: QValue,
    batch: TrajectoryData,
    config: SACConfig,
    key: jax.Array,
) -> jax.Array:
    actions, a_means, a_log_stds = actor_action(
        model_policy,
        batch.states,
        key,
    )
    action_log_densities = norm.logpdf(
        actions, loc=a_means, scale=jnp.exp(a_log_stds)
    ).sum(axis=-1)
    q1_values = model_q1(batch.states, actions)
    q2_values = model_q2(batch.states, actions)
    min_q_values = jnp.minimum(q1_values, q2_values)
    policy_value = jnp.mean(
        min_q_values - config.entropy_coefficient * action_log_densities
    )
    return -policy_value  # Maximize the policy value


def train_step(
    train_states: tuple[
        NNTrainingState, NNTrainingStateSoftTarget, NNTrainingStateSoftTarget
    ],
    batch_and_keys: tuple[TrajectoryData, jax.Array],
    config: SACConfig,
) -> tuple[
    tuple[NNTrainingState, NNTrainingStateSoftTarget, NNTrainingStateSoftTarget],
    tuple[jax.Array, jax.Array, jax.Array],
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
    train_state_policy, train_state_q1, train_state_q2 = train_states
    batch, (key_qvalue, key_policy) = batch_and_keys
    policy = nnx.merge(train_state_policy.model_def, train_state_policy.model_state)
    model_q1 = nnx.merge(train_state_q1.model_def, train_state_q1.model_state)
    model_q1_target = nnx.merge(
        train_state_q1.model_def, train_state_q1.target_model_state
    )
    model_q2 = nnx.merge(train_state_q2.model_def, train_state_q2.model_state)
    model_q2_target = nnx.merge(
        train_state_q2.model_def, train_state_q2.target_model_state
    )

    q_grad_fn = jax.value_and_grad(qvalue_loss_fn, argnums=0)
    q_targets = compute_q_target(
        policy, model_q1_target, model_q2_target, batch, config, key_qvalue
    )
    loss_q1, grads_q1 = q_grad_fn(model_q1, batch, q_targets)
    train_state_q1 = train_state_q1.apply_gradients(grads=grads_q1)
    model_q1 = nnx.merge(train_state_q1.model_def, train_state_q1.model_state)
    loss_q2, grads_q2 = q_grad_fn(model_q2, batch, q_targets)
    train_state_q2 = train_state_q2.apply_gradients(grads=grads_q2)
    model_q2 = nnx.merge(train_state_q2.model_def, train_state_q2.model_state)

    policy_grad_fn = jax.value_and_grad(policy_loss_fn, argnums=0)
    loss_policy, grads_policy = policy_grad_fn(
        policy, model_q1, model_q2, batch, config, key_policy
    )
    train_state_policy = train_state_policy.apply_gradients(grads=grads_policy)

    return (train_state_policy, train_state_q1, train_state_q2), (
        loss_policy,
        loss_q1,
        loss_q2,
    )
