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

    env_name: str
    lr_policy: float
    lr_schedule_policy: str
    lr_end_value_policy: float
    lr_decay_policy: float
    optimizer_policy: str
    lr_qvalue: float
    lr_schedule_qvalue: str
    lr_end_value_qvalue: float
    lr_decay_qvalue: float
    optimizer_qvalue: str
    discount_factor: float
    entropy_coefficient: float
    n_actors: int
    n_actor_steps: int
    batch_size: int
    num_epochs: int
    save_checkpoints: int
    evaluation_frequency: int
    max_training_loops: int
    replay_buffer_size: int
    min_buffer_steps: int
    target_network_update_weight: float

    def to_dict(self) -> dict: ...


def get_config(
    env_name: str = "Hopper-v5",
    lr_policy: float = 3e-4,
    lr_schedule_policy: str = "linear",
    lr_end_value_policy: float = 0.0,
    lr_decay_policy: float = 0.99,
    optimizer_policy: str = "adam",
    lr_qvalue: float = 3e-4,
    lr_schedule_qvalue: str = "linear",
    lr_end_value_qvalue: float = 0.0,
    lr_decay_qvalue: float = 0.99,
    optimizer_qvalue: str = "adam",
    discount_factor: float = 0.99,
    entropy_coefficient: float = 0.2,
    n_actors: int = 10,
    n_actor_steps: int = 128,
    batch_size: int = 256,
    num_epochs: int = 3,
    save_checkpoints: int = 500,
    evaluation_frequency: int = 25,
    max_training_loops: int = 1000000,
    replay_buffer_size: int = 100000,
    min_buffer_steps: int = 5000,
    target_network_update_weight: float = 0.005,
) -> SACConfig:
    """Generates the default configuration for the SAC agent.

    This function provides a base configuration with sensible defaults.
    Also serves as a template to manually create sets of SAC parameters.

    Args:
        env_name: The name of the Gymnasium environment to train on.
        lr_policy: The learning rate for the policy optimizer.
        lr_schedule_policy: The learning rate schedule type for the policy
            ('constant', 'linear', or 'exponential').
        lr_end_value_policy: The end value for the policy learning rate schedule.
        lr_decay_policy: The decay rate for the policy learning rate schedule.
        optimizer_policy: The optimizer to use for policy training.
        lr_qvalue: The learning rate for the Q-value optimizer.
        lr_schedule_qvalue: The learning rate schedule type for the Q-value function
            ('constant', 'linear', or 'exponential').
        lr_end_value_qvalue: The end value for the Q-value learning rate schedule.
        lr_decay_qvalue: The decay rate for the Q-value learning rate schedule.
        optimizer_qvalue: The optimizer to use for Q-value function training.
        discount_factor: The discount factor for future rewards (gamma).
        entropy_coefficient: The coefficient for the entropy term in the loss function.
        n_actors: The number of parallel actors collecting experience.
        n_actor_steps: The number of steps each actor takes before updating the model.
        batch_size: The number of samples per batch for training.
        num_epochs: The number of epochs for training on each batch of data.
        save_checkpoints: Frequency of saving model checkpoints.
        evaluation_frequency: Frequency of running policy evaluations.
        max_training_loops: The maximum number of training loops to perform.
        replay_buffer_size: The maximum size of the replay buffer.
        min_buffer_steps: The minimum number of steps in the replay buffer
            before training.
        target_network_update_weight: The weight for soft updates of the target network.

    Returns:
        A `FrozenConfigDict` containing the default SAC hyperparameters.
    """
    if n_actors * n_actor_steps < batch_size:
        raise ValueError(
            f"Batch size {batch_size} cannot be larger than total "
            f"collected steps {n_actors * n_actor_steps}."
        )
    if optimizer_policy.lower() not in {"adam", "sgd"}:
        raise ValueError(
            f"Unsupported policy optimizer, expected 'adam' or 'sgd'; "
            f"received: {optimizer_policy}"
        )
    if optimizer_qvalue.lower() not in {"adam", "sgd"}:
        raise ValueError(
            f"Unsupported Q-value optimizer, expected 'adam' or 'sgd'; "
            f"received: {optimizer_qvalue}"
        )
    if lr_schedule_policy.lower() not in {"constant", "linear", "exponential"}:
        raise ValueError(
            f"Unsupported policy learning rate schedule, "
            f"expected 'constant', 'linear', or 'exponential'; "
            f"received {lr_schedule_policy}"
        )
    if lr_schedule_qvalue.lower() not in {"constant", "linear", "exponential"}:
        raise ValueError(
            f"Unsupported Q-value learning rate schedule, "
            f"expected 'constant', 'linear', or 'exponential'; "
            f"received {lr_schedule_qvalue}"
        )
    config = make_config(
        {
            "env_name": env_name,
            "lr_policy": lr_policy,
            "lr_schedule_policy": lr_schedule_policy.lower(),
            "lr_end_value_policy": lr_end_value_policy,
            "lr_decay_policy": lr_decay_policy,
            "optimizer_policy": optimizer_policy.lower(),
            "lr_qvalue": lr_qvalue,
            "lr_schedule_qvalue": lr_schedule_qvalue.lower(),
            "lr_end_value_qvalue": lr_end_value_qvalue,
            "lr_decay_qvalue": lr_decay_qvalue,
            "optimizer_qvalue": optimizer_qvalue.lower(),
            "discount_factor": discount_factor,
            "entropy_coefficient": entropy_coefficient,
            "n_actors": n_actors,
            "n_actor_steps": n_actor_steps,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "save_checkpoints": save_checkpoints,
            "evaluation_frequency": evaluation_frequency,
            "max_training_loops": max_training_loops,
            "replay_buffer_size": replay_buffer_size,
            "min_buffer_steps": min_buffer_steps,
            "target_network_update_weight": target_network_update_weight,
        }
    )
    return cast("SACConfig", config)
