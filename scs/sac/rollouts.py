from __future__ import annotations

from typing import (
    TYPE_CHECKING,
)

import jax.numpy as jnp
import numpy as np

from scs.sac.agent import actor_action

if TYPE_CHECKING:
    from flax import nnx
    import gymnasium as gym

    from scs.buffer import ReplayBuffer
    from scs.sac.models import Policy


def sample_transition_to_buffer(
    observation: np.ndarray,
    policy_model: Policy,
    envs: gym.vector.SyncVectorEnv,
    replay_buffer: ReplayBuffer,
    rngs: nnx.Rngs,
) -> tuple[np.ndarray, ReplayBuffer]:
    """Collects a single transition and stores it in the replay buffer.
    The next observation is returned for continuing the episode.
    """
    action, _a_mean, _a_log_std = actor_action(
        policy_model,
        jnp.asarray(observation, dtype=jnp.float32),
        rngs.action_select(),
    )
    next_observation, reward, terminal, truncated, _info = envs.step(  # type: ignore[var-annotated]
        np.tanh(np.asarray(action))
    )
    replay_buffer.add_transitions(
        observation,
        np.asarray(action),
        reward,
        next_observation,
        terminal,
    )
    reset_mask = np.logical_or(terminal, truncated)
    if reset_mask.any():
        observation = envs.reset(options={"reset_mask": reset_mask})[0]
    else:
        observation = next_observation
    return observation, replay_buffer
