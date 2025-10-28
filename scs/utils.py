from __future__ import annotations

from functools import partial
from typing import (
    Hashable,
    Literal,
    overload,
)

import jax
import jax.numpy as jnp
from ml_collections import config_dict


@overload
def make_config(
    config: dict[str, Hashable], frozen: Literal[True] = True
) -> config_dict.FrozenConfigDict: ...


@overload
def make_config(
    config: dict[str, Hashable], frozen: Literal[False]
) -> config_dict.ConfigDict: ...


def make_config(
    config: dict[str, Hashable], frozen: bool = True
) -> config_dict.FrozenConfigDict | config_dict.ConfigDict:
    """Creates a config dict from a built-in python dictionary.

    This function converts a standard Python dictionary into an `ml_collections`
    config dict, which allows for attribute-style access to keys.

    Args:
        config: The input dictionary to be converted. Its keys must be strings,
            and the values can be any hashable type.
        frozen: If True (the default), creates an immutable `FrozenConfigDict`.
            If False, creates a mutable `ConfigDict`.

    Returns:
        An instance of `config_dict.FrozenConfigDict` or
        `config_dict.ConfigDict`.
    """
    if frozen:
        return config_dict.FrozenConfigDict(config)
    return config_dict.ConfigDict(config)


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


@partial(jax.jit, static_argnums=(1,))
def get_expected_return(
    rewards: jax.Array,
    gamma: float,
) -> jax.Array:
    """Computes the discounted returns from a sequence of rewards.

    This function calculates the cumulative discounted rewards for each timestep.

    Note:
        This implementation may face numerical instability with a large number
        of steps. For improved stability, consider using `jax.lax.scan`.

    Args:
        rewards: An array of rewards.
        gamma: The discount factor for future rewards.

    Returns:
        An array of discounted returns.
    """
    discounts = (gamma ** jnp.arange(rewards.shape[0])).reshape(
        (rewards.shape[0],) + (1,) * (rewards.ndim - 1)
    )  # Match dimension of discount to rewards which can be (n,) or (n, m)
    discounted_rewards = rewards * discounts
    returns = jnp.cumsum(discounted_rewards[::-1], axis=0)[::-1] / discounts
    return returns


@partial(jax.jit, static_argnums=(1, 2))
def gae_from_td_residuals(
    td_residuals: jax.Array,
    gamma: float,
    lmbda: float,
) -> jax.Array:
    """Computes Generalized Advantage Estimation (GAE) from TD residuals.

    GAE is used to reduce the variance of advantage estimates, which can help
    stabilize and accelerate policy gradient methods.

    Note:
        This implementation may face numerical instability with a large number
        of steps. For improved stability, consider using `jax.lax.scan`.

    Args:
        td_residuals: The temporal difference errors for each timestep.
        gamma: The discount factor for future rewards.
        lmbda: The lambda parameter for GAE, balancing bias and variance.

    Returns:
        The calculated Generalized Advantage Estimates.
    """
    weighted_discounts = (gamma * lmbda) ** jnp.arange(td_residuals.shape[0])
    discounted_td_residuals = td_residuals * weighted_discounts
    gae = jnp.cumsum(discounted_td_residuals[::-1])[::-1] / weighted_discounts
    return gae
