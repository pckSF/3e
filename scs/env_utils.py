from __future__ import annotations

from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
)

import jax
import jax.numpy as jnp
from mujoco_playground import registry

if TYPE_CHECKING:
    from mujoco_playground import (
        MjxEnv,
        State,
    )


def make_vectorized_env(
    env_name: str,
) -> tuple[
    MjxEnv,
    Callable[[State, jax.Array], State],
    Callable[[State, jax.Array, jax.Array], State],
    Callable[[jax.Array], State],
]:
    """Instantiates an environment and vectorized step and reset functions.

    Args:
        env_name: Identifier understood by `mujoco_playground.registry`.

    Returns:
        A tuple of (raw env, batched step fn, conditional reset fn, batched
        reset fn) ready to plug into JAX-compiled training loops.
    """
    env = registry.load(env_name)

    @jax.vmap(in_axes=(0, 0, 0))
    def _reset_branch(state: State, mask: jax.Array, keys: jax.Array) -> State:
        """Resets the environments whose `mask` entry evaluates to True."""
        return jax.lax.cond(
            mask,
            lambda k: env.reset(k),
            lambda _: state,
            keys,
        )

    def condition_reset(
        env_state: State,
        reset_mask: jax.Array,
        key: jax.Array,
    ) -> State:
        """Reset only the environments marked in `reset_mask`.

        This avoids redundant reset calls by guarding the vmapped branch with a
        single `jnp.any` check.
        """
        reset_mask = reset_mask.astype(bool)
        return jax.lax.cond(
            jnp.any(reset_mask),
            lambda _: _reset_branch(env_state, reset_mask, key),
            lambda _: env_state,
            operand=None,
        )  # Skip resetting when every env is running.

    return (
        env,
        jax.vmap(env.step, in_axes=(0, 0)),
        condition_reset,
        jax.vmap(env.reset),
    )
