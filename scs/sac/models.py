from __future__ import annotations

from typing import TYPE_CHECKING

from flax import nnx

if TYPE_CHECKING:
    import jax


class QValue(nnx.Module):
    def __init__(self, rngs: nnx.Rngs) -> None:
        self.linear_1: nnx.Linear = nnx.Linear(
            in_features=11,
            out_features=256,
            kernel_init=nnx.initializers.orthogonal(),
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
            kernel_init=nnx.initializers.orthogonal(),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.layernorm_2: nnx.LayerNorm = nnx.LayerNorm(
            num_features=256,
            rngs=rngs,
        )
        self.linear_3: nnx.Linear = nnx.Linear(
            in_features=256,
            out_features=128,
            kernel_init=nnx.initializers.orthogonal(),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.layernorm_3: nnx.LayerNorm = nnx.LayerNorm(
            num_features=128,
            rngs=rngs,
        )
        self.qvalue: nnx.Linear = nnx.Linear(
            in_features=128,
            out_features=3,
            kernel_init=nnx.initializers.orthogonal(),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Computes qvalue estimates for states-action pairs.

        Returns:
            - The state-action pair qvalue estimates.
        """
        v = nnx.relu(self.layernorm_1(self.linear_1(x)))
        v = nnx.relu(self.layernorm_2(self.linear_2(v)))
        v = nnx.relu(self.layernorm_3(self.linear_3(v)))

        return self.qvalue(v)


class Policy(nnx.Module):
    def __init__(self, rngs: nnx.Rngs) -> None:
        self.linear_1: nnx.Linear = nnx.Linear(
            in_features=11 + 3,
            out_features=256,
            kernel_init=nnx.initializers.orthogonal(),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.layernorm_1: nnx.LayerNorm = nnx.LayerNorm(
            num_features=256,
            rngs=rngs,
        )
        self.linear_2: nnx.Linear = nnx.Linear(
            in_features=256,
            out_features=256,
            kernel_init=nnx.initializers.orthogonal(),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.layernorm_2: nnx.LayerNorm = nnx.LayerNorm(
            num_features=256,
            rngs=rngs,
        )
        self.linear_3: nnx.Linear = nnx.Linear(
            in_features=256,
            out_features=128,
            kernel_init=nnx.initializers.orthogonal(),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.layernorm_3: nnx.LayerNorm = nnx.LayerNorm(
            num_features=128,
            rngs=rngs,
        )
        self.policy_mean: nnx.Linear = nnx.Linear(
            in_features=128,
            out_features=3,
            kernel_init=nnx.initializers.orthogonal(),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.policy_log_std: nnx.Linear = nnx.Linear(
            in_features=128,
            out_features=3,
            kernel_init=nnx.initializers.orthogonal(),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Computes action distribution parameters for states.

        Returns:
            A tuple containing:
            - The action means.
            - The action log standard deviations.
        """
        p = nnx.relu(self.layernorm_1(self.linear_1(x)))
        p = nnx.relu(self.layernorm_2(self.linear_2(p)))
        p = nnx.relu(self.layernorm_3(self.linear_3(p)))

        return (
            self.policy_mean(p),
            self.policy_log_std(p),
        )
