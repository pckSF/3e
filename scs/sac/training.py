from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from scs.buffer import ReplayBuffer
from scs.evaluation import evaluation_trajectory
from scs.sac.agent import train_step
from scs.sac.rollouts import sample_transition_to_buffer
from scs.utils import get_train_batch_indices

if TYPE_CHECKING:
    import gymnasium as gym

    from scs.data import TrajectoryData
    from scs.data_logging import DataLogger
    from scs.nn_modules import (
        NNTrainingState,
        NNTrainingStateSoftTarget,
    )
    from scs.sac.defaults import SACConfig


@partial(jax.jit, static_argnums=(2,))
def update_on_batches(
    train_state_policy: NNTrainingState,
    train_state_q1: NNTrainingStateSoftTarget,
    train_state_q2: NNTrainingStateSoftTarget,
    batch: TrajectoryData,
    config: SACConfig,
    key: jax.Array,
) -> tuple[
    tuple[NNTrainingState, NNTrainingStateSoftTarget, NNTrainingStateSoftTarget],
    tuple[jax.Array, jax.Array, jax.Array],
]:
    keys = jax.random.split(key, (batch.states.shape[0], 2))
    (
        (train_state_policy, train_state_q1, train_state_q2),
        (loss_policy, loss_q1, loss_q2),
    ) = jax.lax.scan(
        partial(
            train_step,
            config=config,
        ),
        (train_state_policy, train_state_q1, train_state_q2),
        (batch, keys),
    )
    return (train_state_policy, train_state_q1, train_state_q2), (
        loss_policy,
        loss_q1,
        loss_q2,
    )


def train_agent(
    train_state_policy: NNTrainingState,
    train_state_q1: NNTrainingStateSoftTarget,
    train_state_q2: NNTrainingStateSoftTarget,
    envs: gym.vector.SyncVectorEnv,
    config: SACConfig,
    data_logger: DataLogger,
    max_training_loops: int,
    rngs: nnx.Rngs,
) -> tuple[
    tuple[NNTrainingState, NNTrainingStateSoftTarget, NNTrainingStateSoftTarget],
    gym.vector.SyncVectorEnv,
    jax.Array,
    jax.Array,
]:
    """Trains a SAC agent over a specified number of training loops.

    For each loop a trajectory of `config.n_actor_steps` is collected from the
    vectorized environment and used for training.v

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
    replay_buffer: ReplayBuffer = ReplayBuffer(
        max_size=config.replay_buffer_size,
        state_shape=11,
        action_dim=3,
    )
    state: np.ndarray = envs.reset()[0]
    policy_model = nnx.merge(
        train_state_policy.model_def, train_state_policy.model_state
    )
    policy_loss_history: list[float] = []
    mean_q_loss_history: list[float] = []
    eval_history: list[float] = []
    eval_envs: gym.vector.SyncVectorEnv = deepcopy(envs)
    progress_bar: tqdm = tqdm(range(max_training_loops), desc="Training Loops")
    for training_loop in progress_bar:
        state, replay_buffer = sample_transition_to_buffer(
            state=state,
            policy_model=policy_model,
            envs=envs,
            replay_buffer=replay_buffer,
            rng=rngs,
        )
        if replay_buffer.full:
            max_index = replay_buffer.size
        else:
            max_index = replay_buffer.current_index
        batch_indices = get_train_batch_indices(
            n_batches=config.num_epochs,
            batch_size=config.batch_size,
            max_index=max_index,
            resample=False,
            key=rngs.sample(),
        )
        batches = replay_buffer.sample_batch(batch_indices)
        (train_state_policy, train_state_q1, train_state_q2), losses = (
            update_on_batches(
                train_state_policy,
                train_state_q1,
                train_state_q2,
                batches,
                config,
                rngs.action_select(),
            )
        )
        loss_policy, loss_q1, loss_q2 = losses

        if (training_loop + 1) % config.save_checkpoints == 0:
            data_logger.save_checkpoint(
                filename="checkpoint",
                data=train_state_policy.model_state,
            )
        data_logger.save_csv_row("losses_policy", loss_policy)
        data_logger.save_csv_row("losses_q1", loss_q1)
        data_logger.save_csv_row("losses_q2", loss_q2)
        policy_loss_history.append(float(np.mean(loss_policy)))
        mean_q_loss_history.append(float(np.mean(loss_q1 + loss_q2)))
        policy_model = nnx.merge(
            train_state_policy.model_def, train_state_policy.model_state
        )
        if training_loop % config.evaluation_frequency == 0:
            eval_rewards = evaluation_trajectory(
                model=policy_model,
                envs=eval_envs,
                config=config,
                rng=rngs,
            )
            data_logger.save_csv_row("eval_rewards", eval_rewards)
            eval_history.append(float(np.mean(eval_rewards)))
        progress_bar.set_postfix(
            {
                "p_loss": f"{policy_loss_history[-1]:.4f}",
                "q_loss": f"{mean_q_loss_history[-1]:.4f}",
                "eval reward": f"{eval_history[-1]:.2f}",
            }
        )
    data_logger.wait_until_finished()
    return (
        (train_state_policy, train_state_q1, train_state_q2),
        envs,
        jnp.array(policy_loss_history),
        jnp.array(mean_q_loss_history),
        jnp.array(eval_history),
    )
