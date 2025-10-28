from __future__ import annotations

from typing import (
    TYPE_CHECKING,
)

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

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
    rng: nnx.Rngs,
    config: PPOConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    n_envs = envs.num_envs
    max_steps = int(config.n_actor_steps)
    n_envs = int(config.n_actors)
    states = np.zeros((max_steps, n_envs, 8), dtype=np.float32)
    rewards = np.zeros((max_steps, n_envs), dtype=np.float32)
    actions = np.zeros((max_steps, n_envs), dtype=np.uint32)
    next_states = np.zeros((max_steps, n_envs, 8), dtype=np.float32)
    terminals = np.zeros((max_steps, n_envs), dtype=np.bool_)
    episode_mask = np.ones((max_steps, n_envs), dtype=np.float32)

    state = envs.reset()[0]
    terminated = np.zeros(n_envs, dtype=np.bool_)
    for ts in range(max_steps):
        action_logits, _value = model(jnp.asarray(state, dtype=jnp.float32))
        action = jax.random.categorical(rng.action_select(), action_logits)
        next_state, reward, terminal, truncated, _info = envs.step(np.asarray(action))
        states[ts] = state
        next_states[ts] = next_state
        actions[ts] = action
        rewards[ts] = reward
        terminals[ts] = terminal
        reset_env = np.logical_or(terminal, truncated)
        if reset_env.any():
            state = envs.reset(options={"reset_mask": reset_env})[0]
        else:
            state = next_state
        terminated = np.logical_or(terminated, reset_env)
        episode_mask[ts] = terminated
        if terminated.all():
            break
    return (
        jnp.asarray(states, dtype=jnp.float32),
        jnp.asarray(rewards, dtype=jnp.float32),
        jnp.asarray(actions, dtype=jnp.uint32),
        jnp.asarray(next_states, dtype=jnp.float32),
        jnp.asarray(terminals, dtype=jnp.uint32),
        jnp.asarray(np.logical_not(episode_mask), dtype=jnp.float32),
    )
