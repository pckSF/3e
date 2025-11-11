from __future__ import annotations

from typing import TYPE_CHECKING

from flax import nnx
import jax.numpy as jnp

if TYPE_CHECKING:
    import jax


class PolicyValue(nnx.Module):
    def __init__(self, rngs: nnx.Rngs) -> None:
        # Value network layers
        self.value_linear_1: nnx.Linear = nnx.Linear(
            in_features=11,
            out_features=256,
            kernel_init=nnx.initializers.orthogonal(),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.value_layernorm_1: nnx.LayerNorm = nnx.LayerNorm(
            num_features=256,
            rngs=rngs,
        )
        self.value_linear_2 = nnx.Linear(
            in_features=256,
            out_features=256,
            kernel_init=nnx.initializers.orthogonal(),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.value_layernorm_2: nnx.LayerNorm = nnx.LayerNorm(
            num_features=256,
            rngs=rngs,
        )
        self.value_linear_3: nnx.Linear = nnx.Linear(
            in_features=256,
            out_features=128,
            kernel_init=nnx.initializers.orthogonal(),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.value_layernorm_3: nnx.LayerNorm = nnx.LayerNorm(
            num_features=128,
            rngs=rngs,
        )
        self.value: nnx.Linear = nnx.Linear(
            in_features=128,
            out_features=1,
            kernel_init=nnx.initializers.orthogonal(),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )

        # Policy network layers
        self.policy_linear_1: nnx.Linear = nnx.Linear(
            in_features=11,
            out_features=256,
            kernel_init=nnx.initializers.orthogonal(),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.policy_layernorm_1: nnx.LayerNorm = nnx.LayerNorm(
            num_features=256,
            rngs=rngs,
        )
        self.policy_linear_2: nnx.Linear = nnx.Linear(
            in_features=256,
            out_features=256,
            kernel_init=nnx.initializers.orthogonal(),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.policy_layernorm_2: nnx.LayerNorm = nnx.LayerNorm(
            num_features=256,
            rngs=rngs,
        )
        self.policy_linear_3: nnx.Linear = nnx.Linear(
            in_features=256,
            out_features=128,
            kernel_init=nnx.initializers.orthogonal(),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.policy_layernorm_3: nnx.LayerNorm = nnx.LayerNorm(
            num_features=128,
            rngs=rngs,
        )
        self.policy_mean: nnx.Linear = nnx.Linear(
            in_features=128,
            out_features=3,
            kernel_init=nnx.initializers.orthogonal(scale=0.01),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.policy_log_std: nnx.Linear = nnx.Linear(
            in_features=128,
            out_features=3,
            kernel_init=nnx.initializers.orthogonal(scale=0.01),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )

    def __call__(self, state: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Computes action distribution parameters and value estimates for states.

        Returns:
            A tuple containing:
            - The action means.
            - The action log standard deviations.
            - The state value estimates.
        """
        v = nnx.relu(self.value_layernorm_1(self.value_linear_1(state)))
        v = nnx.relu(self.value_layernorm_2(self.value_linear_2(v)))
        v = nnx.relu(self.value_layernorm_3(self.value_linear_3(v)))

        p = nnx.relu(self.policy_layernorm_1(self.policy_linear_1(state)))
        p = nnx.relu(self.policy_layernorm_2(self.policy_linear_2(p)))
        p = nnx.relu(self.policy_layernorm_3(self.policy_linear_3(p)))

        return (
            self.policy_mean(p),
            self.policy_log_std(p),
            self.value(v),
        )

    @nnx.jit
    def get_values(self, states: jax.Array) -> jax.Array:
        """Computes value estimates for the given states."""
        _a_means, _a_log_stds, values = self(states)
        return jnp.squeeze(values)
