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
from scs.evaluation import evaluation_trajectory
from scs.ppo.agent import (
    BatchLossMetrics,
    train_step,
)
from scs.ppo.rollouts import collect_trajectories

if TYPE_CHECKING:
    import gymnasium as gym

    from scs.data_logging import DataLogger
    from scs.nn_modules import NNTrainingState
    from scs.ppo.defaults import PPOConfig


@partial(jax.jit, static_argnums=(2,))
def update_on_trajectory(
    train_state: NNTrainingState,
    trajectories: TrajectoryData,
    config: PPOConfig,
    key: jax.Array,
) -> tuple[NNTrainingState, jax.Array, BatchLossMetrics]:
    """Performs multiple updates on one collected trajectory.

    Computes advantages and samples `config.num_epochs` batch indices for the
    provided trajectory and performs training updates on the agent's model for
    each batch.

    Args:
        train_state: The neural network model's training state container.
        trajectories: Trajectories of experience collected from the environment.
        config: The agent's configuration.
        key: A JAX random key for generating batch indices.

    Returns:
        A tuple containing the updated training state and the loss values for
        each training step.
    """
    trajectory_advantages = compute_advantages(
        trajectories,
        nnx.merge(train_state.model_def, train_state.model_state),
        config,
    )
    trajectories = stack_agent_trajectories(trajectories)
    trajectory_advantages = stack_agent_advantages(trajectory_advantages)

    total_steps = trajectories.n_steps
    num_minibatches = total_steps // config.batch_size
    epoch_keys = jax.random.split(key, config.num_epochs)

    def run_epoch(
        carry_state: NNTrainingState, epoch_key: jax.Array
    ) -> tuple[NNTrainingState, tuple[jax.Array, BatchLossMetrics]]:
        permutation = jax.random.permutation(epoch_key, total_steps)
        minibatch_indices = permutation.reshape((num_minibatches, config.batch_size))
        return jax.lax.scan(
            partial(
                train_step,
                trajectory=trajectories,
                trajectory_advantages=trajectory_advantages,
                config=config,
            ),
            carry_state,
            minibatch_indices,
        )

    train_state, (epoch_losses, epoch_metrics) = jax.lax.scan(
        run_epoch,
        train_state,
        epoch_keys,
    )

    mean_loss = jnp.mean(epoch_losses)
    mean_metrics = jax.tree_util.tree_map(jnp.mean, epoch_metrics)
    return train_state, mean_loss, mean_metrics


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
        rngs: A container for flax.nnx random number generators.

    Returns:
        A tuple containing the final training state, the environment, and arrays
        of the loss and evaluation histories.
    """
    data_logger.store_metadata("config", config.to_dict())
    steps_per_rollout = int(config.n_actor_steps * config.n_actors)
    if steps_per_rollout % config.batch_size != 0:
        raise ValueError(
            "PPO batch_size must evenly divide the total number of collected "
            f"samples per rollout. Got batch_size={config.batch_size} with "
            f"steps_per_rollout={steps_per_rollout}."
        )
    states: np.ndarray = envs.reset()[0]
    reset_mask = np.zeros((config.n_actors,), dtype=bool)
    loss_history: list[float] = []
    eval_history: list[float] = []
    eval_envs: gym.vector.SyncVectorEnv = deepcopy(envs)
    progress_bar: tqdm = tqdm(range(max_training_loops), desc="Training Loops")
    for training_loop in progress_bar:
        trajectories, reset_mask, states = collect_trajectories(
            model=nnx.merge(train_state.model_def, train_state.model_state),
            envs=envs,
            reset_mask=reset_mask,
            state=states,
            config=config,
            rng=rngs,
        )
        train_state, loss, loss_metrics = update_on_trajectory(
            train_state=train_state,
            trajectories=trajectories,
            config=config,
            key=rngs.sample(),
        )
        if (training_loop + 1) % config.save_checkpoints == 0:
            data_logger.save_checkpoint(
                filename="checkpoint",
                data=train_state.model_state,
            )
        data_logger.save_csv_row("losses", loss)
        data_logger.save_csv_row("ppo_value", loss_metrics.ppo_value)
        data_logger.save_csv_row("value_loss", loss_metrics.value_loss)
        data_logger.save_csv_row("entropy", loss_metrics.entropy)
        data_logger.save_csv_row("kl_estimate", loss_metrics.kl_estimate)
        loss_history.append(float(np.asarray(loss)))
        if training_loop % config.evaluation_frequency == 0:
            eval_rewards = evaluation_trajectory(
                model=nnx.merge(train_state.model_def, train_state.model_state),
                envs=eval_envs,
                config=config,
                rng=rngs,
            )
            data_logger.save_csv_row("eval_rewards", eval_rewards)
            eval_history.append(float(np.mean(eval_rewards)))
        latest_eval = eval_history[-1] if eval_history else float("nan")
        progress_bar.set_postfix(
            {
                "loss": f"{loss_history[-1]:.4f}",
                "eval reward": f"{latest_eval:.2f}",
                "kl": f"{float(np.asarray(loss_metrics.kl_estimate)):.4f}",
            }
        )
    data_logger.wait_until_finished()
    return train_state, envs, jnp.array(loss_history), jnp.array(eval_history)
