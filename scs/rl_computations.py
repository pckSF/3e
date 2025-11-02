from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from scs.data import TrajectoryData


@partial(jax.jit, static_argnums=(2,))
def masked_standardize(
    x: jax.Array, mask: jax.Array, epsilon: float = 1e-5
) -> jax.Array:
    """Standardizes an array using a mask, ignoring padded values.

    In reinforcement learning, it's common to pad episode data to a fixed
    length for batch processing. This function standardizes the data (e.g.,
    returns or advantages) while correctly handling the padded elements by
    using a mask.

    Args:
        x: The array to be standardized.
        mask: A binary array where `1` indicates valid data and `0` indicates
            padding.
        epsilon: A small value added to the standard deviation for numerical
            stability.

    Returns:
        The standardized array, with padded elements remaining as zero.
    """
    sum_elements = jnp.sum(mask, axis=0, keepdims=True)
    mean = jnp.sum(x * mask, axis=0, keepdims=True) / sum_elements
    variance = jnp.sum(((x - mean) * mask) ** 2 * mask) / sum_elements
    std = jnp.sqrt(variance) + epsilon
    return ((x - mean) / std) * mask


def _get_expected_return(
    previous_return: jax.Array,
    trajectory_step: TrajectoryData,
    gamma: float,
) -> tuple[jax.Array, jax.Array]:
    """Computes one step of the discounted return; has to be run in reverse!"""
    reward, terminal = trajectory_step.rewards, trajectory_step.terminals
    current_return = reward + gamma * previous_return * (1.0 - terminal)
    return current_return, current_return


@partial(jax.jit, static_argnums=(1,))
def calculate_expected_return(
    trajectory: TrajectoryData,
    gamma: float,
) -> jax.Array:
    """Computes the discounted returns for a trajectory.

    Terminal states reset the discounted sum to account for episode endings and
    the environment restarting within a trajectory.

    Args:
        trajectory: A `TrajectoryData` object containing rewards and terminal
            flags.
        gamma: The discount factor for future rewards.

    Returns:
        An array of discounted returns for each step in the trajectory.
    """
    _, expected_returns = jax.lax.scan(
        partial(_get_expected_return, gamma=gamma),
        jnp.zeros_like(trajectory.rewards[-1]),
        trajectory,
        reverse=True,
    )
    return expected_returns


def _get_gae_value(
    previous_gae: jax.Array,
    residual_terminal: tuple[jax.Array, jax.Array],
    gamma: float,
    lmbda: float,
) -> tuple[jax.Array, jax.Array]:
    """Computes one step of the GAE; has to be run in reverse!"""
    td_residual, terminal = residual_terminal
    current_gae = td_residual + gamma * lmbda * previous_gae * (1.0 - terminal)
    return current_gae, current_gae


@partial(jax.jit, static_argnums=(1, 2))
def gae_from_td_residuals(
    td_residuals: jax.Array,
    terminals: jax.Array,
    gamma: float,
    lmbda: float,
) -> jax.Array:
    """Computes Generalized Advantage Estimation (GAE) from TD residuals.

    Terminal states reset the discounted sum to account for episode endings and
    the environment restarting within a trajectory.

    Args:
        td_residuals: The temporal difference errors for each timestep.
        terminals: A boolean array indicating the end of an episode.
        gamma: The discount factor for future rewards.
        lmbda: The lambda parameter for GAE, balancing bias and variance.

    Returns:
        The calculated Generalized Advantage Estimates for each timestep.
    """
    _, gae = jax.lax.scan(
        partial(_get_gae_value, gamma=gamma, lmbda=lmbda),
        jnp.zeros_like(td_residuals[-1]),
        (td_residuals, terminals),
        reverse=True,
    )
    return gae


@partial(
    jax.jit,
    static_argnums=(
        4,
        5,
    ),
)
def calculate_gae(
    rewards: jax.Array,
    values: jax.Array,
    next_values: jax.Array,
    terminals: jax.Array,
    gamma: float,
    lmbda: float,
) -> jax.Array:
    td_error = rewards + gamma * next_values * (1.0 - terminals) - values
    return gae_from_td_residuals(td_error, terminals, gamma, lmbda)
