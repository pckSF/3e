from __future__ import annotations

from typing import (
    TYPE_CHECKING,
)

import jax.numpy as jnp
import numpy as np

from scs.ppo.agent import actor_action

if TYPE_CHECKING:
    from flax import nnx
    import gymnasium as gym

    from scs.ppo.defaults import PPOConfig
    from scs.ppo.models import PolicyValue
    from scs.sac.defaults import SACConfig
    from scs.sac.models import Policy


def evaluation_trajectory(
    model: Policy | PolicyValue,
    envs: gym.vector.SyncVectorEnv,
    rng: nnx.Rngs,
    config: PPOConfig | SACConfig,
) -> np.ndarray:
    """Runs the agent for a full evaluation trajectory in parallel environments.

    This function evaluates the agent's performance by running it until all
    parallel episodes terminate or a maximum step limit is reached. It
    accumulates the total reward for each episode.

    Args:
        model: The model to be evaluated.
        envs: The vectorized `gymnasium` environment.
        rng: The JAX random number generator state.
        config: The PPO or SAC configuration object.

    Returns:
        A NumPy array containing the final cumulative reward for each
        parallel environment.
    """
    n_envs = int(config.n_actors)
    rewards = np.zeros((n_envs,), dtype=np.float32)

    state: np.ndarray = envs.reset()[0]
    terminated = np.zeros((n_envs,), dtype=bool)
    for _ts in range(10000):
        action, _a_mean, _a_log_std = actor_action(
            model,
            jnp.asarray(state, dtype=jnp.float32),
            rng,
        )
        state, step_reward, terminal, truncated, _info = envs.step(  # type: ignore[var-annotated]
            np.tanh(np.asarray(action))
        )

        rewards += step_reward * np.logical_not(terminated)

        reset_mask = np.logical_or(terminal, truncated)
        terminated = np.logical_or(terminated, terminal)
        if reset_mask.any():
            # Required to avoid error raised when passing an action to a terminated
            # environment. TODO: Better way to handle this?
            state = envs.reset(options={"reset_mask": reset_mask})[0]
        if terminated.all():
            break
    return rewards
