from __future__ import annotations

from typing import (
    TYPE_CHECKING,
)

from flax import nnx
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from scs.data_logging import DataLogger
    from scs.ppo.defaults import PPOConfig
    from scs.ppo.models import PolicyValue
    from scs.sac.defaults import SACConfig
    from scs.sac.models import Policy


@nnx.jit
def actor_action(
    model_policy: Policy | PolicyValue,
    states: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Samples an action from an actor's policy."""
    a_means, a_log_stds = model_policy(states)[:2]
    actions = a_means + jnp.exp(a_log_stds) * jax.random.normal(
        key, shape=a_means.shape
    )
    return actions, a_means, a_log_stds


def evaluation_trajectory(
    model: Policy | PolicyValue,
    envs: gym.vector.SyncVectorEnv,
    config: PPOConfig | SACConfig,
    rng: nnx.Rngs,
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
            rng.action_select(),
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


def render_trajectory(
    model: Policy | PolicyValue,
    config: PPOConfig | SACConfig,
    data_logger: DataLogger,
    rng: nnx.Rngs,
    max_steps: int = 1000,
) -> None:
    """Renders the agent interacting with the environment and saves to video file.

    Args:
        model: The model to be evaluated.
        config: The PPO or SAC configuration object.
        rng: The JAX random number generator state.
        data_logger: The data logger instance for managing output directories.
        max_steps: The maximum number of steps to render.

    Returns:
        The total reward accumulated during the episode.
    """
    render_dir = data_logger.log_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    video_numbers = (
        int(p.stem.split("_")[1].split("-")[0])
        for p in render_dir.glob("render_*.mp4")
        if p.stem.split("_")[1].split("-")[0].isdigit()
    )
    count = max(video_numbers, default=0) + 1

    env = gym.make(config.env_name, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(render_dir),
        name_prefix=f"render_{count:05d}",
        episode_trigger=lambda _: True,  # Record every episode
    )

    state: np.ndarray = env.reset()[0]
    total_reward = 0.0

    for _ts in range(max_steps):
        action, _a_mean, _a_log_std = actor_action(
            model,
            jnp.asarray(state, dtype=jnp.float32),
            rng.action_select(),
        )
        state, step_reward, terminal, truncated, _info = env.step(
            np.tanh(np.asarray(action))
        )
        total_reward += float(step_reward)

        if terminal or truncated:
            break

    env.close()
    data_logger.log_info(
        f"Rendered video saved: render_{count:05d} (reward: {total_reward:.2f})"
    )
