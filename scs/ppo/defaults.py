from __future__ import annotations

from typing import (
    Protocol,
    cast,
)

from scs.configuration import make_config


class PPOConfig(Protocol):
    """Defines the structure of a PPO configuration object for static analysis.

    This protocol ensures that any configuration object used with the PPO agent
    will have the required hyperparameters, enabling static type checking and
    improving code completion.
    """

    learning_rate: float
    learning_rate_end_value: float
    learning_rate_decay: float
    optimizer: str
    lr_schedule: str
    discount_factor: float
    clip_parameter: float
    entropy_coefficient: float
    gae_lambda: float
    n_actors: int
    n_actor_steps: int
    batch_size: int
    num_epochs: int
    value_loss_coefficient: float
    exploration_coefficient: float
    save_checkpoints: int
    evaluation_frequency: int
    normalize_advantages: bool
    max_training_loops: int

    def to_dict(self) -> dict: ...


def get_config(
    learning_rate: float = 2.5e-4,
    learning_rate_end_value: float = 0.0,
    learning_rate_decay: float = 0.99,
    optimizer: str = "adam",
    lr_schedule: str = "linear",
    discount_factor: float = 0.99,
    clip_parameter: float = 0.1,
    entropy_coefficient: float = 0.01,
    gae_lambda: float = 0.95,
    n_actors: int = 10,
    n_actor_steps: int = 128,
    batch_size: int = 256,
    num_epochs: int = 3,
    value_loss_coefficient: float = 0.5,
    exploration_coefficient: float = 0.01,
    save_checkpoints: int = 500,
    evaluation_frequency: int = 25,
    normalize_advantages: bool = False,
    max_training_loops: int = 10000,
) -> PPOConfig:
    """Generates the default configuration for the PPO agent.

    This function provides a base configuration with sensible defaults.
    Also serves as a template to manually create sets of PPO parameters.

    Args:
        learning_rate: The learning rate for the Adam optimizer.
        learning_rate_end_value: The end value for the learning rate schedule.
        learning_rate_decay: The decay rate for the learning rate schedule.
        optimizer: The optimizer to use for training.
        lr_schedule: The learning rate schedule type.
        discount_factor: The discount factor for future rewards (gamma).
        clip_parameter: The clipping parameter for the PPO surrogate objective.
        entropy_coefficient: The coefficient for the entropy term in the loss.
        gae_lambda: The lambda parameter for Generalized Advantage Estimation (GAE).
        n_actors: The number of parallel actors collecting experience.
        n_actor_steps: The number of steps each actor takes before updating the model.
        batch_size: The number of samples per batch for training.
        num_epochs: The number of epochs for training on each batch of data.
        value_loss_coefficient: Coefficient for the value loss term.
        exploration_coefficient: Coefficient for the exploration bonus.
        save_checkpoints: Frequency of saving model checkpoints.
        evaluation_frequency: Frequency of running policy evaluations.
        normalize_advantages: Whether to normalize advantages.
        max_training_loops: The maximum number of training loops to perform.

    Returns:
        A `FrozenConfigDict` containing the default PPO hyperparameters.
    """
    if n_actors * n_actor_steps < batch_size:
        raise ValueError(
            f"Batch size {batch_size} cannot be larger than total "
            f"collected steps {n_actors * n_actor_steps}."
        )
    if optimizer.lower() not in ("adam", "sgd"):
        raise ValueError(
            f"Unsupported optimizer, expected 'adam' or 'sgd'; received: {optimizer}"
        )
    if lr_schedule.lower() not in ("linear", "exponential"):
        raise ValueError(
            f"Unsupported learning rate schedule, expected 'linear' or "
            f"'exponential'; received {lr_schedule}"
        )
    config = make_config(
        {
            "learning_rate": learning_rate,
            "learning_rate_end_value": learning_rate_end_value,
            "learning_rate_decay": learning_rate_decay,
            "optimizer": optimizer.lower(),
            "lr_schedule": lr_schedule.lower(),
            "discount_factor": discount_factor,
            "clip_parameter": clip_parameter,
            "entropy_coefficient": entropy_coefficient,
            "gae_lambda": gae_lambda,
            "n_actors": n_actors,
            "n_actor_steps": n_actor_steps,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "value_loss_coefficient": value_loss_coefficient,
            "exploration_coefficient": exploration_coefficient,
            "save_checkpoints": save_checkpoints,
            "evaluation_frequency": evaluation_frequency,
            "normalize_advantages": normalize_advantages,
            "max_training_loops": max_training_loops,
        }
    )
    return cast("PPOConfig", config)
