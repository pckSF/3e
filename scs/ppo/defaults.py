from __future__ import annotations

from typing import TYPE_CHECKING

from scs.utils import make_config

if TYPE_CHECKING:
    from ml_collections.config_dict import FrozenConfigDict


def get_config(
    learning_rate: float = 2.5e-4,
    discount_factor: float = 0.99,
    clip_parameter: float = 0.1,
    entropy_coefficient: float = 0.01,
    gae_lambda: float = 0.95,
    batch_size: int = 64,
    num_epochs: int = 5,
) -> FrozenConfigDict:
    """Generates the default configuration for the PPO agent.

    This function provides a base configuration with sensible defaults.
    Also serves as a template to manually create sets of PPO parameters.

    Args:
        learning_rate: The learning rate for the Adam optimizer.
        discount_factor: The discount factor for future rewards (gamma).
        clip_parameter: The clipping parameter for the PPO surrogate objective.
        entropy_coefficient: The coefficient for the entropy term in the loss.
        gae_lambda: The lambda parameter for Generalized Advantage Estimation (GAE).
        batch_size: The number of samples per batch for training.
        num_epochs: The number of epochs for training on each batch of data.

    Returns:
        A `FrozenConfigDict` containing the default PPO hyperparameters.
    """
    return make_config(
        {
            "learning_rate": learning_rate,
            "discount_factor": discount_factor,
            "clip_parameter": clip_parameter,
            "entropy_coefficient": entropy_coefficient,
            "gae_lambda": gae_lambda,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
        }
    )
