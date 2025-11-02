from __future__ import annotations

from typing import (
    TYPE_CHECKING,
)

import jax.numpy as jnp
import numpy as np

from scs import utils
from scs.data import TrajectoryData
from scs.ppo.agent import actor_action
from scs.rl_computations import calculate_gae

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
            - A `TrajectoryData` object holding the collected states, actions,
            rewards, next_states, and terminal signals.
            - The final `reset_mask`, indicating which environments terminated
            during this collection phase, to be used in the next call.
    """
    n_envs = int(config.n_actors)
    max_steps = int(config.n_actor_steps)

    states = np.zeros((max_steps, n_envs, 11), dtype=np.float32)
    rewards = np.zeros((max_steps, n_envs), dtype=np.float32)
    actions = np.zeros((max_steps, n_envs, 3), dtype=np.uint32)
    next_states = np.zeros((max_steps, n_envs, 11), dtype=np.float32)
    terminals = np.zeros((max_steps, n_envs), dtype=np.bool_)
    values = np.zeros((max_steps, n_envs), dtype=np.float32)

    if reset_mask.any():
        continue_state: np.ndarray = envs.reset(options={"reset_mask": reset_mask})[0]
        # TODO: Remove this check once it works reliably
        utils.states_healthcheck(state, continue_state, np.logical_not(reset_mask))
        state = continue_state

    for ts in range(max_steps):
        action, a_noise, _a_mean, _a_log_std, value = actor_action(
            model,
            jnp.asarray(state, dtype=jnp.float32),
            rng,
            config,
        )
        next_state, reward, terminal, truncated, _info = envs.step(  # type: ignore[var-annotated]
            np.tanh(np.asarray(action + a_noise))
        )

        states[ts] = state
        next_states[ts] = next_state
        actions[ts] = action + a_noise
        rewards[ts] = reward
        terminals[ts] = terminal
        values[ts] = np.asarray(value, dtype=np.float32)

        reset_mask = np.logical_or(terminal, truncated)
        if reset_mask.any():
            state = envs.reset(options={"reset_mask": reset_mask})[0]
        else:
            state = next_state

    next_values = model(jnp.asarray(next_states))[2]
    gae = calculate_gae(
        rewards=jnp.asarray(rewards, dtype=jnp.float32),
        values=jnp.asarray(values, dtype=jnp.float32),
        next_values=next_values,
        terminals=jnp.asarray(terminals, dtype=jnp.float32),
        gamma=config.discount_factor,
        lmbda=config.gae_lambda,
    )

    return (
        TrajectoryData(
            states=jnp.asarray(states, dtype=jnp.float32),
            actions=jnp.asarray(actions, dtype=jnp.uint32),
            rewards=jnp.asarray(rewards, dtype=jnp.float32),
            next_states=jnp.asarray(next_states, dtype=jnp.float32),
            terminals=jnp.asarray(terminals, dtype=jnp.uint32),
            values=jnp.asarray(values, dtype=jnp.float32),
            gae=gae,
            n_steps=max_steps,
            agents=jnp.arange(n_envs),
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
        action, _a_noise, _a_mean, _a_log_std, _value = actor_action(
            model,
            jnp.asarray(state, dtype=jnp.float32),
            rng,
            config,
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
            state = envs.reset(options={"reset_mask": terminated})[0]
        if terminated.all():
            break
    return rewards
