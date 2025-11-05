from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from scs.data import (
    TrajectoryData,
    compute_advantages,
    stack_agent_advantages,
    stack_agent_trajectories,
)
from scs.ppo.agent import train_step
from scs.ppo.rollouts import (
    collect_trajectories,
    evaluation_trajectory,
)
from scs.utils import get_train_batch_indices

if TYPE_CHECKING:
    import gymnasium as gym

    from scs.data_logging import DataLogger
    from scs.nn_modules import NNTrainingState
    from scs.ppo.defaults import PPOConfig


@partial(jax.jit, static_argnums=(2,))
def update_on_trajectory(
    train_state: NNTrainingState,
    trajectory: TrajectoryData,
    config: PPOConfig,
    key: jax.Array,
) -> tuple[
    NNTrainingState, jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array]
]:
    """Performs multiple updates on one collected trajectory.

    Computes advantages and samples `config.num_epochs` batch indices for the
    provided trajectory and performs training updates on the agent's model for
    each batch.

    Args:
        train_state: The neural network model's training state container.
        trajectory: A trajectory of experience collected from the environment.
        config: The agent's configuration.
        key: A JAX random key for generating batch indices.

    Returns:
        A tuple containing the updated training state and the loss values for
        each training step.
    """
    trajectory_advantages = compute_advantages(
        trajectory,
        nnx.merge(train_state.model_def, train_state.model_state),
        config,
    )
    trajectories = stack_agent_trajectories(trajectory)
    trajectory_advantages = stack_agent_advantages(trajectory_advantages)
    batch_indices = get_train_batch_indices(
        n_batches=config.num_epochs,
        batch_size=config.batch_size,
        max_index=trajectories.n_steps,
        resample=False,
        key=key,
    )
    train_state, (loss, loss_components) = jax.lax.scan(
        partial(
            train_step,
            trajectory=trajectories,
            trajectory_advantages=trajectory_advantages,
            config=config,
        ),
        train_state,
        batch_indices,
    )
    return train_state, loss, loss_components


def train_agent(
    train_state: NNTrainingState,
    envs: gym.vector.SyncVectorEnv,
    config: PPOConfig,
    data_logger: DataLogger,
    max_training_loops: int,
    rngs: nnx.Rngs,
) -> tuple[NNTrainingState, gym.vector.SyncVectorEnv, jax.Array, jax.Array]:
    """Trains a PPO agent over a specified number of training loops.

    For each loop a trajectory of `config.n_actor_steps` is collected from the
    vectorized environment and used for training.

    Evaluation is done on a copy of the environment.
    TODO: Is this even necessary?

    Args:
        train_state: The initial state of the neural network model.
        envs: The vectorized gym environment for training.
        config: The agent's configuration.
        data_logger: A logger for saving training data and model checkpoints.
        max_training_loops: The total number of training loops to execute.
        rngs: A container for JAX random number generators.

    Returns:
        A tuple containing the final training state, the environment, and arrays
        of the loss and evaluation histories.
    """
    data_logger.store_metadata("config", config.to_dict())
    states: np.ndarray = envs.reset()[0]
    reset_mask = np.zeros((config.n_actors,), dtype=bool)
    model = nnx.merge(train_state.model_def, train_state.model_state)
    loss_history: list[float] = []
    eval_history: list[float] = []
    eval_envs: gym.vector.SyncVectorEnv = deepcopy(envs)
    progress_bar: tqdm = tqdm(range(max_training_loops), desc="Training Loops")
    for training_loop in progress_bar:
        trajectories, reset_mask, states = collect_trajectories(
            model=model,
            envs=envs,
            reset_mask=reset_mask,
            state=states,
            rng=rngs,
            config=config,
        )
        train_state, loss, loss_components = update_on_trajectory(
            train_state=train_state,
            trajectory=trajectories,
            config=config,
            key=rngs.sample(),
        )
        if (training_loop + 1) % config.save_checkpoints == 0:
            data_logger.save_checkpoint(
                filename="checkpoint",
                data=train_state.model_state,
            )
        data_logger.save_csv_row("losses", loss)
        ppo_value, value_loss, entropy, kl_estimate = loss_components[0]
        data_logger.save_csv_row("ppo_value", ppo_value)
        data_logger.save_csv_row("value_loss", value_loss)
        data_logger.save_csv_row("entropy", entropy)
        data_logger.save_csv_row("kl_estimate", kl_estimate)
        loss_history.append(float(np.mean(loss)))
        model = nnx.merge(train_state.model_def, train_state.model_state)
        if training_loop % config.evaluation_frequency == 0:
            eval_rewards = evaluation_trajectory(
                model=model,
                envs=eval_envs,
                config=config,
                rng=rngs,
            )
            data_logger.save_csv_row("eval_rewards", eval_rewards)
            eval_history.append(float(np.mean(eval_rewards)))
        progress_bar.set_postfix(
            {
                "loss": f"{loss_history[-1]:.4f}",
                "eval reward": f"{eval_history[-1]:.2f}",
            }
        )
    return train_state, envs, jnp.array(loss_history), jnp.array(eval_history)
