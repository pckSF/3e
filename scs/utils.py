from __future__ import annotations

from functools import partial
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


@partial(jax.jit, static_argnums=(0, 1, 2, 4))
def get_train_batch_indices(
    samples: int,
    batch_size: int,
    max_index: int,
    key: jax.Array,
    replace_for_rows: bool = False,
) -> jax.Array:
    """Generates random indices for training batches.

    These indices can be used to slice out a batch from trajectory data.
    """
    indices = jnp.arange(max_index)
    if replace_for_rows:
        return jax.random.choice(key, indices, (samples, batch_size), replace=True)
    else:
        keys = jax.random.split(key, samples)
        return jax.lax.map(
            partial(jax.random.choice, a=indices, shape=(batch_size,), replace=False),
            keys,
        )
