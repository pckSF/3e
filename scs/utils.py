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


@partial(
    jax.jit,
    static_argnums=(
        0,
        1,
        2,
        3,
    ),
)
def get_train_batch_indices(
    n_batches: int,
    batch_size: int,
    max_index: int,
    resample: bool,
    key: jax.Array,
) -> jax.Array:
    """Generates random indices for training batches.

    Without resampling `n_batches * batch_size` must be less or equal than the
    `max_index` since no sample is allowed to repeat across the set of batches.

    With resampling, each batch is sampled independently, allowing for samples
    to reappear in multiple batches.
    """
    indices = jnp.arange(max_index)
    if not resample:
        return jax.random.choice(key, indices, (n_batches, batch_size), replace=False)
    else:
        keys = jax.random.split(key, n_batches)
        return jax.lax.map(
            partial(jax.random.choice, a=indices, shape=(batch_size,), replace=False),
            keys,
        )
