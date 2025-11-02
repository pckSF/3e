from __future__ import annotations

from typing import TYPE_CHECKING

from flax import nnx

if TYPE_CHECKING:
    import jax


class ActorCritic(nnx.Module):
    def __init__(self, rngs: nnx.Rngs) -> None:
        self.linear_1: nnx.Linear = nnx.Linear(
            in_features=11,
            out_features=256,
            kernel_init=nnx.initializers.glorot_uniform(),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.layernorm_1: nnx.LayerNorm = nnx.LayerNorm(
            num_features=256,
            rngs=rngs,
        )
        self.linear_2 = nnx.Linear(
            in_features=256,
            out_features=256,
            kernel_init=nnx.initializers.glorot_uniform(),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.layernorm_2: nnx.LayerNorm = nnx.LayerNorm(
            num_features=256,
            rngs=rngs,
        )
        self.critic_linear_1: nnx.Linear = nnx.Linear(
            in_features=256,
            out_features=128,
            kernel_init=nnx.initializers.glorot_uniform(),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.critic_layernorm_1: nnx.LayerNorm = nnx.LayerNorm(
            num_features=128,
            rngs=rngs,
        )
        self.critic: nnx.Linear = nnx.Linear(
            in_features=128,
            out_features=1,
            kernel_init=nnx.initializers.glorot_uniform(),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.actor_linear_1: nnx.Linear = nnx.Linear(
            in_features=256,
            out_features=128,
            kernel_init=nnx.initializers.glorot_uniform(),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.actor_layernorm_1: nnx.LayerNorm = nnx.LayerNorm(
            num_features=128,
            rngs=rngs,
        )
        self.actor_mean: nnx.Linear = nnx.Linear(
            in_features=128,
            out_features=3,
            kernel_init=nnx.initializers.glorot_uniform(),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.actor_log_std: nnx.Linear = nnx.Linear(
            in_features=128,
            out_features=3,
            kernel_init=nnx.initializers.glorot_uniform(),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        x = nnx.relu(self.layernorm_1(self.linear_1(x)))
        x = nnx.relu(self.layernorm_2(self.linear_2(x)))
        c = nnx.relu(self.critic_layernorm_1(self.critic_linear_1(x)))
        a = nnx.relu(self.actor_layernorm_1(self.actor_linear_1(x)))
        return (
            self.actor_mean(a),
            self.actor_log_std(a),
            self.critic(c),
        )

    @nnx.jit
    def get_values(self, states: jax.Array) -> jax.Array:
        """Computes value estimates for the given states."""
        _a_means, _a_log_stds, values = self(states)
        return values
