from __future__ import annotations

from typing import (
    TYPE_CHECKING,
)

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from scs import utils
from scs.data import TrajectoryData

if TYPE_CHECKING:
    import gymnasium as gym

    from scs.ppo.defaults import PPOConfig
    from scs.ppo.models import ActorCritic


@nnx.jit(static_argnums=(3,))
def actor_action(
    model: ActorCritic,
    states: jax.Array,
    rng: nnx.Rngs,
    config: PPOConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Samples an action from the actor's policy.

    This function computes the action distribution from the actor model, samples
    an action, and adds exploration noise.
    """
    a_means, a_log_stds, _values = model(states)
    noise = jax.random.normal(rng.noise(), (config.n_actors,)) * config.action_noise
    actions = a_means + jnp.exp(a_log_stds) * jax.random.normal(
        rng.action(), (config.n_actors,)
    )
    return actions + noise, a_means, a_log_stds


def collect_trajectories(
    model: ActorCritic,
    envs: gym.vector.SyncVectorEnv,
    reset_mask: np.ndarray,
    state: np.ndarray,
    rng: nnx.Rngs,
    config: PPOConfig,
) -> tuple[TrajectoryData, np.ndarray]:
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
    n_envs = envs.num_envs
    max_steps = int(config.n_actor_steps)

    n_envs = int(config.n_actors)
    states = np.zeros((max_steps, n_envs, 11), dtype=np.float32)
    rewards = np.zeros((max_steps, n_envs), dtype=np.float32)
    actions = np.zeros((max_steps, n_envs, 3), dtype=np.uint32)
    next_states = np.zeros((max_steps, n_envs, 11), dtype=np.float32)
    terminals = np.zeros((max_steps, n_envs), dtype=np.bool_)

    if reset_mask.any():
        continue_state = envs.reset(options={"reset_mask": reset_mask})[0]
        utils.states_healthcheck(state, continue_state, reset_mask)
        state = continue_state

    for ts in range(max_steps):
        action, _a_means, _a_log_stds = actor_action(
            model,
            jnp.asarray(state, dtype=jnp.float32),
            rng,
            config,
        )
        next_state, reward, terminal, truncated, _info = envs.step(np.asarray(action))

        states[ts] = state
        next_states[ts] = next_state
        actions[ts] = action
        rewards[ts] = reward
        terminals[ts] = terminal

        reset_mask = np.logical_or(terminal, truncated)
        if reset_mask.any():
            state = envs.reset(options={"reset_mask": reset_mask})[0]
        else:
            state = next_state

    return TrajectoryData(
        states=jnp.asarray(states, dtype=jnp.float32),
        actions=jnp.asarray(actions, dtype=jnp.uint32),
        rewards=jnp.asarray(rewards, dtype=jnp.float32),
        next_states=jnp.asarray(next_states, dtype=jnp.float32),
        terminals=jnp.asarray(terminals, dtype=jnp.uint32),
        n_steps=max_steps,
        agents=jnp.arange(n_envs),
    ), reset_mask
