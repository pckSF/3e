from __future__ import annotations

from typing import TYPE_CHECKING

from flax import nnx

if TYPE_CHECKING:
    import jax


class ActorCritic(nnx.Module):
    def __init__(self, rngs: nnx.Rngs) -> None:
        self.linear_1: nnx.Linear = nnx.Linear(
            in_features=4,
            out_features=128,
            kernel_init=nnx.initializers.glorot_uniform(),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.actor: nnx.Linear = nnx.Linear(
            in_features=128,
            out_features=2,
            kernel_init=nnx.initializers.glorot_uniform(),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.citic: nnx.Linear = nnx.Linear(
            in_features=128,
            out_features=1,
            kernel_init=nnx.initializers.glorot_uniform(),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = nnx.relu(self.linear_1(x))
        return self.actor(x), self.citic(x)
