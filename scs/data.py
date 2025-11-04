from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

from flax import struct
import jax.numpy as jnp

from scs.rl_computations import calculate_gae

if TYPE_CHECKING:
    import jax

    from scs.ppo.defaults import PPOConfig
    from scs.ppo.models import ActorCritic


@struct.dataclass
class TrajectoryData:
    """JAX-friendly PyTree data container.

    Fields that are scanned over axis 0:
    - states:               [T, N, ...]
    - actions:              [T, N, ...]
    - action_log_densities: [T, N, ...]
    - rewards:              [T, N]
    - next_states:          [T, N, ...]
    - terminals:            [T, N]

    Static metadata:
    - n_steps:              int
    - agents:               int
    - samples:              bool
    """

    states: jax.Array
    actions: jax.Array
    action_log_densities: jax.Array
    rewards: jax.Array
    next_states: jax.Array
    terminals: jax.Array
    n_steps: int = struct.field(pytree_node=False)
    agents: int = struct.field(pytree_node=False)
    samples: bool = struct.field(pytree_node=False, default=False)

    @classmethod
    def load(cls, path: str) -> TrajectoryData:
        """Loads TrajectoryData from a pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(**data)

    def save(self, path: str) -> None:
        """Saves TrajectoryData to a pickle file."""
        data_dict = {
            "states": self.states,
            "actions": self.actions,
            "action_log_densities": self.action_log_densities,
            "rewards": self.rewards,
            "next_states": self.next_states,
            "terminals": self.terminals,
            "n_steps": self.n_steps,
            "agents": self.agents,
            "samples": self.samples,
        }
        with open(path, "wb") as f:
            pickle.dump(data_dict, f)


def stack_agent_trajectories(data: TrajectoryData) -> TrajectoryData:
    """Stacks the agent dimension into the batch dimension.

    Returns:
        A TrajectoryData object with shape [T * N, ...] for states, actions,
        action_log_densities, rewards, next_states, and terminals.
    """
    steps, agents = data.n_steps, data.agents
    return TrajectoryData(
        states=data.states.reshape((steps * agents, *data.states.shape[2:])),
        actions=data.actions.reshape((steps * agents, *data.actions.shape[2:])),
        action_log_densities=data.action_log_densities.reshape(
            (steps * agents, *data.action_log_densities.shape[2:])
        ),
        rewards=data.rewards.reshape((steps * agents,)),
        next_states=data.next_states.reshape(
            (steps * agents, *data.next_states.shape[2:])
        ),
        terminals=data.terminals.reshape((steps * agents,)),
        n_steps=steps * agents,
        agents=1,
        samples=data.samples,
    )


def get_trajectory_batch(
    data: TrajectoryData, batch_indices: jax.Array
) -> TrajectoryData:
    """Returns a batch of data based on the provided indices."""
    return TrajectoryData(
        states=data.states[batch_indices],
        actions=data.actions[batch_indices],
        action_log_densities=data.action_log_densities[batch_indices],
        rewards=data.rewards[batch_indices],
        next_states=data.next_states[batch_indices],
        terminals=data.terminals[batch_indices],
        n_steps=batch_indices.shape[0],
        agents=data.agents,
        samples=True,
    )


@struct.dataclass
class TrajectoryGAE:
    """JAX-friendly PyTree data container that value predictions and the GAE.

    Fields that are scanned over axis 0:
    - values:           [T, N]
    - advantages:       [T, N]

    Static metadata:
    - n_steps:         int
    - agents:          int
    - gamma:           float
    - lam:             float
    - samples:         bool
    """

    values: jax.Array
    advantages: jax.Array
    n_steps: int = struct.field(pytree_node=False)
    agents: int = struct.field(pytree_node=False)
    gamma: float = struct.field(pytree_node=False)
    lam: float = struct.field(pytree_node=False)
    samples: bool = struct.field(pytree_node=False, default=False)

    @classmethod
    def load(cls, path: str) -> TrajectoryGAE:
        """Loads TrajectoryData from a pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(**data)

    def save(self, path: str) -> None:
        """Saves TrajectoryGAE to a pickle file."""
        data_dict = {
            "values": self.values,
            "advantages": self.advantages,
            "n_steps": self.n_steps,
            "agents": self.agents,
            "gamma": self.gamma,
            "lam": self.lam,
            "samples": self.samples,
        }
        with open(path, "wb") as f:
            pickle.dump(data_dict, f)


def compute_advantages(
    trajectory: TrajectoryData,
    model: ActorCritic,
    config: PPOConfig,
) -> TrajectoryGAE:
    """Creates a TrajectoryGAE object from a standard trajectory.


    Important: This function assumes trajectory.samples=False.
    Using sampled/shuffled trajectories will produce incorrect results
    as GAE requires temporal ordering.

    Args:
        trajectory: The base trajectory data.
        model: The actor-critic model to compute values.
        config: The PPO configuration containing gamma and lambda.

    Returns:
        A new TrajectoryGAE object with GAE and value data.
    """
    values = model.get_values(trajectory.states)
    next_values = model.get_values(trajectory.next_states)
    # TODO: Normalize Advantages?
    advantages = calculate_gae(
        rewards=trajectory.rewards,
        values=values,
        next_values=next_values,
        terminals=trajectory.terminals,
        gamma=config.discount_factor,
        lmbda=config.gae_lambda,
    )

    if config.normalize_advantages:
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

    return TrajectoryGAE(
        values=values,
        advantages=advantages,
        n_steps=trajectory.n_steps,
        agents=trajectory.agents,
        gamma=config.discount_factor,
        lam=config.gae_lambda,
    )


def stack_agent_advantages(data: TrajectoryGAE) -> TrajectoryGAE:
    """Stacks the agent dimension into the batch dimension.

    Returns:
        A TrajectoryGAE object with shape [T * N, ...] for values and advantages.
    """
    steps, agents = data.n_steps, data.agents
    return TrajectoryGAE(
        values=data.values.reshape((steps * agents,)),
        advantages=data.advantages.reshape((steps * agents,)),
        n_steps=steps * agents,
        agents=1,
        gamma=data.gamma,
        lam=data.lam,
        samples=data.samples,
    )


def get_advantage_batch(data: TrajectoryGAE, batch_indices: jax.Array) -> TrajectoryGAE:
    """Returns a batch of data based on the provided indices."""
    return TrajectoryGAE(
        values=data.values[batch_indices],
        advantages=data.advantages[batch_indices],
        n_steps=data.n_steps,
        agents=data.agents,
        gamma=data.gamma,
        lam=data.lam,
        samples=True,
    )
