from __future__ import annotations

from typing import (
    TYPE_CHECKING,
)

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
import numpy as np

from scs.data import (
    TrajectoryData,
)
from scs.ppo.agent import actor_action

if TYPE_CHECKING:
    from flax import nnx
    import gymnasium as gym

    from scs.ppo.defaults import PPOConfig
    from scs.ppo.models import ActorCritic


def collect_trajectories(
    model: ActorCritic,
    envs: gym.vector.SyncVectorEnv,
    reset_mask: np.ndarray,
    state: np.ndarray,
    rng: nnx.Rngs,
    config: PPOConfig,
) -> tuple[TrajectoryData, np.ndarray, np.ndarray]:
    """Collects trajectories by interacting with parallel environments.

    This function runs the agent's policy in a vectorized environment for a
    fixed number of steps (`n_actor_steps`). It does not run full episodes,
    but rather collects a fixed number of parallel timesteps. Individual
    episodes within the vectorized environment may terminate and restart
    during collection. The primary goal is to gather `n_actor_steps` of
    usable interaction data for training.

    Args:
        model: The `ActorCritic` model used to select actions.
        envs: The vectorized `gymnasium` environment for interaction.
        reset_mask: A boolean array indicating which environments in the vector
            should be reset before starting data collection. This is passed from
            the previous collection step to handle episodes that terminated.
        state: The initial state of the environments. This is the last observed
            state from the previous collection step, ensuring continuity.
        rng: The JAX random number generator state.
        config: The PPO configuration object.

    Returns:
        A tuple containing:
            - A `TrajectoryDataPPO` object holding the collected states, actions,
                rewards, next_states, and terminal signals.
            - The final `reset_mask`, indicating which environments terminated
            during this collection phase, to be used in the next call.
    """
    n_envs = int(config.n_actors)
    max_steps = int(config.n_actor_steps)

    states = np.zeros((max_steps, n_envs, 11), dtype=np.float32)
    rewards = np.zeros((max_steps, n_envs), dtype=np.float32)
    actions = np.zeros((max_steps, n_envs, 3), dtype=np.float32)
    action_log_densities = np.zeros((max_steps, n_envs, 3), dtype=np.float32)
    next_states = np.zeros((max_steps, n_envs, 11), dtype=np.float32)
    terminals = np.zeros((max_steps, n_envs), dtype=np.bool_)

    if reset_mask.any():
        state = envs.reset(options={"reset_mask": reset_mask})[0]

    for ts in range(max_steps):
        action, a_mean, a_log_std = actor_action(
            model,
            jnp.asarray(state, dtype=jnp.float32),
            rng,
        )
        action_log_density = jax.jit(norm.logpdf)(
            action,
            loc=a_mean,
            scale=jnp.exp(a_log_std),
        )
        next_state, reward, terminal, truncated, _info = envs.step(  # type: ignore[var-annotated]
            np.tanh(np.asarray(action))
        )

        states[ts] = state
        next_states[ts] = next_state
        actions[ts] = np.asarray(action)
        action_log_densities[ts] = np.asarray(action_log_density)
        rewards[ts] = reward
        terminals[ts] = terminal

        reset_mask = np.logical_or(terminal, truncated)
        if reset_mask.any():
            state = envs.reset(options={"reset_mask": reset_mask})[0]
        else:
            state = next_state

    return (
        TrajectoryData(
            states=jnp.asarray(states, dtype=jnp.float32),
            actions=jnp.asarray(actions, dtype=jnp.float32),
            action_log_densities=jnp.asarray(action_log_densities, dtype=jnp.float32),
            rewards=jnp.asarray(rewards, dtype=jnp.float32),
            next_states=jnp.asarray(next_states, dtype=jnp.float32),
            terminals=jnp.asarray(terminals, dtype=jnp.uint32),
            n_steps=max_steps,
            agents=n_envs,
        ),
        reset_mask,
        state,
    )


def evaluation_trajectory(
    model: ActorCritic,
    envs: gym.vector.SyncVectorEnv,
    rng: nnx.Rngs,
    config: PPOConfig,
) -> np.ndarray:
    """Runs the agent for a full evaluation trajectory in parallel environments.

    This function evaluates the agent's performance by running it until all
    parallel episodes terminate or a maximum step limit is reached. It
    accumulates the total reward for each episode.

    Args:
        model: The `ActorCritic` model to be evaluated.
        envs: The vectorized `gymnasium` environment.
        rng: The JAX random number generator state.
        config: The PPO configuration object.

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
