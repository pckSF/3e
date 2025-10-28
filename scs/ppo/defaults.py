from __future__ import annotations

from typing import (
    Protocol,
    cast,
)

from scs.utils_config import make_config


class PPOConfig(Protocol):
    """Defines the structure of a PPO configuration object for static analysis.

    This protocol ensures that any configuration object used with the PPO agent
    will have the required hyperparameters, enabling static type checking and
    improving code completion.
    """

    learning_rate: float
    discount_factor: float
    clip_parameter: float
    entropy_coefficient: float
    gae_lambda: float
    n_actors: int
    n_actor_steps: int
    batch_size: int
    num_epochs: int
    action_noise: float


def get_config(
    learning_rate: float = 2.5e-4,
    discount_factor: float = 0.99,
    clip_parameter: float = 0.1,
    entropy_coefficient: float = 0.01,
    gae_lambda: float = 0.95,
    n_actors: int = 4,
    n_actor_steps: int = 128,
    batch_size: int = 64,
    num_epochs: int = 5,
    action_noise: float = 0.2,
) -> PPOConfig:
    """Generates the default configuration for the PPO agent.

    This function provides a base configuration with sensible defaults.
    Also serves as a template to manually create sets of PPO parameters.

    Args:
        learning_rate: The learning rate for the Adam optimizer.
        discount_factor: The discount factor for future rewards (gamma).
        clip_parameter: The clipping parameter for the PPO surrogate objective.
        entropy_coefficient: The coefficient for the entropy term in the loss.
        gae_lambda: The lambda parameter for Generalized Advantage Estimation (GAE).
        n_actors: The number of parallel actors collecting experience.
        n_actor_steps: The number of steps each actor takes before updating the model.
        batch_size: The number of samples per batch for training.
        num_epochs: The number of epochs for training on each batch of data.
        action_noise: Noise added to the actions for exploration.

    Returns:
        A `FrozenConfigDict` containing the default PPO hyperparameters.
    """
    config = make_config(
        {
            "learning_rate": learning_rate,
            "discount_factor": discount_factor,
            "clip_parameter": clip_parameter,
            "entropy_coefficient": entropy_coefficient,
            "gae_lambda": gae_lambda,
            "n_actors": n_actors,
            "n_actor_steps": n_actor_steps,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "action_noise": action_noise,
        }
    )
    return cast("PPOConfig", config)
