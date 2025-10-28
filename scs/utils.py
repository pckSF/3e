from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    import numpy as np


def compare_states(
    state1: jax.Array | np.ndarray,
    state2: jax.Array | np.ndarray,
    mask: jax.Array | np.ndarray | None = None,
) -> bool:
    """Compares two state arrays for equality."""
    if mask is not None:
        state1, state2 = state1[mask], state2[mask]
    return bool(jnp.array_equal(state1, state2))


def states_healthcheck(
    state1: jax.Array | np.ndarray,
    state2: jax.Array | np.ndarray,
    mask: jax.Array | np.ndarray,
) -> None:
    if not compare_states(state1, state2, mask):
        raise ValueError(
            f"Relevant state arrays do not match!\n"
            f"State 1: {state1[mask]}\n"
            f"State 2: {state2[mask]}"
        )
