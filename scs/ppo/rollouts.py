from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

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
    from mujoco_playground import State

    from scs.nn_modules import NNTrainingState
    from scs.ppo.defaults import PPOConfig
    from scs.ppo.models import PolicyValue


def _scan_timestep(
    env_state: State,
    key: jax.Array,
    train_state: NNTrainingState,
    step: Callable[[State, jax.Array], State],
    conditional_reset: Callable[[State, jax.Array, jax.Array], State],
) -> tuple[
    State, tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]
]:
    model = nnx.merge(train_state.model_def, train_state.model_state)
    a_mean, a_log_std, _value = model(env_state.obs)
    action = a_mean + jnp.exp(a_log_std) * jax.random.normal(key, shape=a_mean.shape)
    action_log_density = norm.logpdf(
        action,
        loc=a_mean,
        scale=jnp.exp(a_log_std),
    ).sum(axis=-1)
    next_env_state = step(env_state, jnp.tanh(action))
    reset_mask = next_env_state.done.astype(bool)
    env_state = conditional_reset(next_env_state, reset_mask, key)
    return env_state, (
        env_state.obs,
        action,
        action_log_density,
        next_env_state.reward,
        next_env_state.obs,
        next_env_state.done,
    )


def collect_trajectories(
    model: PolicyValue,
    envs: gym.vector.SyncVectorEnv,
    reset_mask: np.ndarray,
    observation: np.ndarray,
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
        model: The `PolicyValue` model used to select actions.
        envs: The vectorized `gymnasium` environment for interaction.
        reset_mask: A boolean array indicating which environments in the vector
            should be reset before starting data collection. This is passed from
            the previous collection step to handle episodes that terminated.
        observation: The initial observation of the environments. This is the
            last observation from the previous collection step, ensuring
            continuity.
        rng: The JAX random number generator state.
        config: The PPO configuration object.

    Returns:
        A tuple containing:
            - A `TrajectoryDataPPO` object holding the collected observations,
                actions, rewards, next_observations, and terminal signals.
            - The final `reset_mask`, indicating which environments terminated
            during this collection phase, to be used in the next call.
    """
    n_envs = int(config.n_actors)
    max_steps = int(config.n_actor_steps)

    observations = np.zeros((max_steps, n_envs, 11), dtype=np.float32)
    rewards = np.zeros((max_steps, n_envs), dtype=np.float32)
    actions = np.zeros((max_steps, n_envs, 3), dtype=np.float32)
    action_log_densities = np.zeros((max_steps, n_envs, 3), dtype=np.float32)
    next_observations = np.zeros((max_steps, n_envs, 11), dtype=np.float32)
    terminals = np.zeros((max_steps, n_envs), dtype=np.bool_)

    if reset_mask.any():
        observation = envs.reset(options={"reset_mask": reset_mask})[0]

    for ts in range(max_steps):
        action, a_mean, a_log_std = actor_action(
            model,
            jnp.asarray(observation, dtype=jnp.float32),
            rng.action_select(),
        )
        action_log_density = jax.jit(norm.logpdf)(
            action,
            loc=a_mean,
            scale=jnp.exp(a_log_std),
        )
        next_observation, reward, terminal, truncated, _info = envs.step(  # type: ignore[var-annotated]
            np.tanh(np.asarray(action))
        )

        observations[ts] = observation
        next_observations[ts] = next_observation
        actions[ts] = np.asarray(action)
        action_log_densities[ts] = np.asarray(action_log_density)
        rewards[ts] = reward
        terminals[ts] = terminal

        reset_mask = np.logical_or(terminal, truncated)
        if reset_mask.any():
            observation = envs.reset(options={"reset_mask": reset_mask})[0]
        else:
            observation = next_observation

    return (
        TrajectoryData(
            observations=jnp.asarray(observations, dtype=jnp.float32),
            actions=jnp.asarray(actions, dtype=jnp.float32),
            action_log_densities=jnp.asarray(action_log_densities, dtype=jnp.float32),
            rewards=jnp.asarray(rewards, dtype=jnp.float32),
            next_observations=jnp.asarray(next_observations, dtype=jnp.float32),
            terminals=jnp.asarray(terminals, dtype=jnp.uint32),
            n_steps=max_steps,
            agents=n_envs,
        ),
        reset_mask,
        observation,
    )
