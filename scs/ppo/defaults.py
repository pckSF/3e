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

    env_name: str
    lr_policyvalue: float
    lr_schedule_policyvalue: str
    lr_end_value_policyvalue: float
    lr_decay_policyvalue: float
    optimizer_policyvalue: str
    discount_factor: float
    clip_parameter: float
    entropy_coefficient: float
    gae_lambda: float
    n_actors: int
    n_actor_steps: int
    batch_size: int
    num_epochs: int
    value_loss_coefficient: float
    max_grad_norm_policyvalue: float
    save_checkpoints: int
    evaluation_frequency: int
    normalize_advantages: bool
    max_training_loops: int

    def to_dict(self) -> dict: ...


def get_config(
    env_name: str = "Hopper-v5",
    lr_policyvalue: float = 2.5e-4,
    lr_schedule_policyvalue: str = "linear",
    lr_end_value_policyvalue: float = 0.0,
    lr_decay_policyvalue: float = 0.99,
    optimizer_policyvalue: str = "adam",
    discount_factor: float = 0.99,
    clip_parameter: float = 0.1,
    entropy_coefficient: float = 0.01,
    gae_lambda: float = 0.95,
    n_actors: int = 10,
    n_actor_steps: int = 128,
    batch_size: int = 256,
    num_epochs: int = 3,
    value_loss_coefficient: float = 0.5,
    max_grad_norm_policyvalue: float = 1.0,
    save_checkpoints: int = 500,
    evaluation_frequency: int = 25,
    normalize_advantages: bool = False,
    max_training_loops: int = 10000,
) -> PPOConfig:
    """Generates the default configuration for the PPO agent.

    This function provides a base configuration with sensible defaults.
    Also serves as a template to manually create sets of PPO parameters.

    Args:
        env_name: The name of the Gymnasium environment to train on.
        lr_policyvalue: The learning rate for the shared policy-value optimizer.
        lr_schedule_policyvalue: The learning rate schedule type
            ('constant', 'linear', or 'exponential').
        lr_end_value_policyvalue: The end value for the learning rate schedule.
        lr_decay_policyvalue: The decay rate for the learning rate schedule.
        optimizer_policyvalue: The optimizer to use for training.
        discount_factor: The discount factor for future rewards (gamma).
        clip_parameter: The clipping parameter for the PPO surrogate objective.
        entropy_coefficient: The coefficient for the entropy term in the loss.
        gae_lambda: The lambda parameter for Generalized Advantage Estimation (GAE).
        n_actors: The number of parallel actors collecting experience.
        n_actor_steps: The number of steps each actor takes before updating the model.
        batch_size: The number of samples per batch for training.
        num_epochs: The number of epochs for training on each batch of data.
        value_loss_coefficient: Coefficient for the value loss term.
    max_grad_norm_policyvalue: Gradient clipping threshold for the shared optimizer.
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
    if optimizer_policyvalue.lower() not in {"adam", "sgd"}:
        raise ValueError(
            f"Unsupported optimizer, expected 'adam' or 'sgd'; "
            f"received: {optimizer_policyvalue}"
        )
    if lr_schedule_policyvalue.lower() not in {"constant", "linear", "exponential"}:
        raise ValueError(
            f"Unsupported learning rate schedule, "
            f"expected 'constant', 'linear', or 'exponential'; "
            f"received {lr_schedule_policyvalue}"
        )
    config = make_config(
        {
            "env_name": env_name,
            "lr_policyvalue": lr_policyvalue,
            "lr_schedule_policyvalue": lr_schedule_policyvalue.lower(),
            "lr_end_value_policyvalue": lr_end_value_policyvalue,
            "lr_decay_policyvalue": lr_decay_policyvalue,
            "optimizer_policyvalue": optimizer_policyvalue.lower(),
            "discount_factor": discount_factor,
            "clip_parameter": clip_parameter,
            "entropy_coefficient": entropy_coefficient,
            "gae_lambda": gae_lambda,
            "n_actors": n_actors,
            "n_actor_steps": n_actor_steps,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "value_loss_coefficient": value_loss_coefficient,
            "max_grad_norm_policyvalue": max_grad_norm_policyvalue,
            "save_checkpoints": save_checkpoints,
            "evaluation_frequency": evaluation_frequency,
            "normalize_advantages": normalize_advantages,
            "max_training_loops": max_training_loops,
        }
    )
    return cast("PPOConfig", config)
