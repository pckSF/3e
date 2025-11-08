from __future__ import annotations

from typing import TYPE_CHECKING

from flax import nnx
import jax.numpy as jnp
import optax

if TYPE_CHECKING:
    import jax

    from scs.ppo.defaults import PPOConfig


def get_optimizer(config: PPOConfig) -> optax.GradientTransformation:
    if config.optimizer == "adam":
        optimizer = optax.adam
    elif config.optimizer == "sgd":
        optimizer = optax.sgd
    else:
        raise ValueError(
            f"Unsupported optimizer, expected 'adam' or 'sgd'; "
            f"received: {config.optimizer}"
        )
    if config.lr_schedule == "linear":
        lr_schedule = optax.linear_schedule(
            init_value=config.learning_rate,
            end_value=config.learning_rate_end_value,
            transition_steps=(config.num_epochs * config.max_training_loops),
        )
    elif config.lr_schedule == "exponential":
        lr_schedule = optax.exponential_decay(
            init_value=config.learning_rate,
            transition_steps=config.num_epochs * config.max_training_loops,
            decay_rate=config.learning_rate_decay,
            end_value=config.learning_rate_end_value,
        )
    else:
        raise ValueError(
            f"Unsupported learning rate schedule, expected 'linear' or "
            f"'exponential'; received {config.lr_schedule}"
        )
    return optimizer(learning_rate=lr_schedule)


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
        self.mean_layernorm_1: nnx.LayerNorm = nnx.LayerNorm(
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

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Computes action distribution parameters and value estimates for states.

        Returns:
            A tuple containing:
            - The action means.
            - The action log standard deviations.
            - The state value estimates.
        """
        v = nnx.relu(self.value_layernorm_1(self.value_linear_1(x)))
        v = nnx.relu(self.value_layernorm_2(self.value_linear_2(v)))
        v = nnx.relu(self.value_layernorm_3(self.value_linear_3(v)))

        p = nnx.relu(self.policy_layernorm_1(self.policy_linear_1(x)))
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
