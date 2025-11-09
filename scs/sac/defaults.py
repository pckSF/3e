from __future__ import annotations

from typing import (
    Protocol,
    cast,
)

from scs.configuration import make_config


class SACConfig(Protocol):
    """Defines the structure of a SAC configuration object for static analysis.

    This protocol ensures that any configuration object used with the SAC agent
    will have the required hyperparameters, enabling static type checking and
    improving code completion.
    """

    lr_policy: float
    lr_schedule_policy: str
    lr_end_value_policy: float
    lr_decay_policy: float
    optimizer_policy: str
    lr_value: float
    lr_schedule_value: str
    lr_end_value_value: float
    lr_decay_value: float
    optimizer_value: str
    discount_factor: float
    n_actors: int
    n_actor_steps: int
    batch_size: int
    num_epochs: int
    save_checkpoints: int
    evaluation_frequency: int
    max_training_loops: int

    def to_dict(self) -> dict: ...


def get_config(
    lr_policy: float = 3e-4,
    lr_schedule_policy: str = "linear",
    lr_end_value_policy: float = 0.0,
    lr_decay_policy: float = 0.99,
    optimizer_policy: str = "adam",
    lr_value: float = 3e-4,
    lr_schedule_value: str = "linear",
    lr_end_value_value: float = 0.0,
    lr_decay_value: float = 0.99,
    optimizer_value: str = "adam",
    discount_factor: float = 0.99,
    n_actors: int = 10,
    n_actor_steps: int = 128,
    batch_size: int = 256,
    num_epochs: int = 3,
    save_checkpoints: int = 500,
    evaluation_frequency: int = 25,
    max_training_loops: int = 10000,
) -> SACConfig:
    """Generates the default configuration for the SAC agent.

    This function provides a base configuration with sensible defaults.
    Also serves as a template to manually create sets of SAC parameters.

    Args:
        lr_policy: The learning rate for the policy optimizer.
        lr_schedule_policy: The learning rate schedule type for the policy.
        lr_end_value_policy: The end value for the policy learning rate schedule.
        lr_decay_policy: The decay rate for the policy learning rate schedule.
        optimizer_policy: The optimizer to use for policy training.
        lr_value: The learning rate for the value optimizer.
        lr_schedule_value: The learning rate schedule type for the value function.
        lr_end_value_value: The end value for the value learning rate schedule.
        lr_decay_value: The decay rate for the value learning rate schedule.
        optimizer_value: The optimizer to use for value function training.
        discount_factor: The discount factor for future rewards (gamma).
        n_actors: The number of parallel actors collecting experience.
        n_actor_steps: The number of steps each actor takes before updating the model.
        batch_size: The number of samples per batch for training.
        num_epochs: The number of epochs for training on each batch of data.
        save_checkpoints: Frequency of saving model checkpoints.
        evaluation_frequency: Frequency of running policy evaluations.
        max_training_loops: The maximum number of training loops to perform.

    Returns:
        A `FrozenConfigDict` containing the default SAC hyperparameters.
    """
    if n_actors * n_actor_steps < batch_size:
        raise ValueError(
            f"Batch size {batch_size} cannot be larger than total "
            f"collected steps {n_actors * n_actor_steps}."
        )
    if optimizer_policy.lower() not in ("adam", "sgd"):
        raise ValueError(
            f"Unsupported policy optimizer, expected 'adam' or 'sgd'; "
            f"received: {optimizer_policy}"
        )
    if optimizer_value.lower() not in ("adam", "sgd"):
        raise ValueError(
            f"Unsupported value optimizer, expected 'adam' or 'sgd'; "
            f"received: {optimizer_value}"
        )
    if lr_schedule_policy.lower() not in ("linear", "exponential"):
        raise ValueError(
            f"Unsupported policy learning rate schedule, expected 'linear' or "
            f"'exponential'; received {lr_schedule_policy}"
        )
    if lr_schedule_value.lower() not in ("linear", "exponential"):
        raise ValueError(
            f"Unsupported value learning rate schedule, expected 'linear' or "
            f"'exponential'; received {lr_schedule_value}"
        )
    config = make_config(
        {
            "lr_policy": lr_policy,
            "lr_schedule_policy": lr_schedule_policy.lower(),
            "lr_end_value_policy": lr_end_value_policy,
            "lr_decay_policy": lr_decay_policy,
            "optimizer_policy": optimizer_policy.lower(),
            "lr_value": lr_value,
            "lr_schedule_value": lr_schedule_value.lower(),
            "lr_end_value_value": lr_end_value_value,
            "lr_decay_value": lr_decay_value,
            "optimizer_value": optimizer_value.lower(),
            "discount_factor": discount_factor,
            "n_actors": n_actors,
            "n_actor_steps": n_actor_steps,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "save_checkpoints": save_checkpoints,
            "evaluation_frequency": evaluation_frequency,
            "max_training_loops": max_training_loops,
        }
    )
    return cast("SACConfig", config)
