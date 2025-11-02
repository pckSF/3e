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
    - states:           [T, N, ...]
    - actions:          [T, N, ...]
    - rewards:          [T, N]
    - next_states:      [T, N, ...]
    - terminals:        [T, N]

    Static metadata:
    - n_steps:         int
    - agents:          [N]
    """

    states: jax.Array
    actions: jax.Array
    rewards: jax.Array
    next_states: jax.Array
    terminals: jax.Array
    n_steps: int = struct.field(pytree_node=False)
    agents: jax.Array = struct.field(pytree_node=False)

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
            "rewards": self.rewards,
            "next_states": self.next_states,
            "terminals": self.terminals,
            "n_steps": self.n_steps,
            "agents": self.agents,
        }
        with open(path, "wb") as f:
            pickle.dump(data_dict, f)

    def get_batch_data(self, batch_indices: jax.Array) -> TrajectoryData:
        """Returns a batch of data based on the provided indices."""
        return TrajectoryData(
            states=self.states[batch_indices],
            actions=self.actions[batch_indices],
            rewards=self.rewards[batch_indices],
            next_states=self.next_states[batch_indices],
            terminals=self.terminals[batch_indices],
            n_steps=batch_indices.shape[0],
            agents=self.agents,
        )  # type: ignore[call-arg]

    def stack_agent_trajectories(self) -> TrajectoryData:
        """Stacks the agent dimension into the batch dimension.

        Returns:
            A TrajectoryData object with shape [T * N, ...] for states, actions,
            rewards, next_states, and terminals.
        """
        steps, agents = self.n_steps, self.agents.shape[0]
        return TrajectoryData(
            states=self.states.reshape((steps * agents, *self.states.shape[2:])),
            actions=self.actions.reshape((steps * agents, *self.actions.shape[2:])),
            rewards=self.rewards.reshape((steps * agents,)),
            next_states=self.next_states.reshape(
                (steps * agents, *self.next_states.shape[2:])
            ),
            terminals=self.terminals.reshape((steps * agents,)),
            n_steps=steps * agents,
            agents=jnp.array(1),
        )  # type: ignore[call-arg]


class TrajectoryDataPPO(TrajectoryData):
    """JAX-friendly PyTree data container that contains additional data for PPO.

    Fields that are scanned over axis 0:
    - states:           [T, N, ...]
    - actions:          [T, N, ...]
    - rewards:          [T, N]
    - next_states:      [T, N, ...]
    - terminals:        [T, N]
    - values:           [T, N]
    - gae:              [T, N]

    Static metadata:
    - n_steps:         int
    - agents:          [N]
    - gamma:           float
    - lam:             float
    """

    values: jax.Array
    gae: jax.Array
    gamma: float = struct.field(pytree_node=False)
    lam: float = struct.field(pytree_node=False)

    @classmethod
    def load(cls, path: str) -> TrajectoryDataPPO:
        """Loads TrajectoryData from a pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(**data)

    def save(self, path: str) -> None:
        """Saves TrajectoryData to a pickle file."""
        data_dict = {
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_states": self.next_states,
            "terminals": self.terminals,
            "n_steps": self.n_steps,
            "agents": self.agents,
            "values": self.values,
            "gae": self.gae,
            "gamma": self.gamma,
            "lam": self.lam,
        }
        with open(path, "wb") as f:
            pickle.dump(data_dict, f)

    @classmethod
    def create_from_trajectory(
        cls,
        trajectory: TrajectoryData,
        model: ActorCritic,
        config: PPOConfig,
    ) -> TrajectoryDataPPO:
        values = model(trajectory.states)[2]
        next_values = model(trajectory.next_states)[2]
        gae = calculate_gae(
            rewards=trajectory.rewards,
            values=values,
            next_values=next_values,
            terminals=trajectory.terminals,
            gamma=config.discount_factor,
            lmbda=config.gae_lambda,
        )
        return cls(
            states=trajectory.states,
            actions=trajectory.actions,
            rewards=trajectory.rewards,
            next_states=trajectory.next_states,
            terminals=trajectory.terminals,
            n_steps=trajectory.n_steps,
            agents=trajectory.agents,
            values=values,
            gae=gae,
            gamma=config.discount_factor,
            lam=config.gae_lambda,
        )

    def update_with_model(
        self,
        model: ActorCritic,
    ) -> TrajectoryDataPPO:
        values = model(self.states)[2]
        next_values = model(self.next_states)[2]
        gae = calculate_gae(
            rewards=self.rewards,
            values=values,
            next_values=next_values,
            terminals=self.terminals,
            gamma=self.gamma,
            lmbda=self.lam,
        )
        return TrajectoryDataPPO(
            states=self.states,
            actions=self.actions,
            rewards=self.rewards,
            next_states=self.next_states,
            terminals=self.terminals,
            n_steps=self.n_steps,
            agents=self.agents,
            values=values,
            gae=gae,
            gamma=self.gamma,
            lam=self.lam,
        )

    def get_batch_data(self, batch_indices: jax.Array) -> TrajectoryDataPPO:
        """Returns a batch of data based on the provided indices."""
        return TrajectoryDataPPO(
            states=self.states[batch_indices],
            actions=self.actions[batch_indices],
            rewards=self.rewards[batch_indices],
            next_states=self.next_states[batch_indices],
            terminals=self.terminals[batch_indices],
            n_steps=batch_indices.shape[0],
            agents=self.agents,
            values=self.values[batch_indices],
            gae=self.gae[batch_indices],
            gamma=self.gamma,
            lam=self.lam,
        )  # type: ignore[call-arg]

    def stack_agent_trajectories(self) -> TrajectoryDataPPO:
        """Stacks the agent dimension into the batch dimension.

        Returns:
            A TrajectoryData object with shape [T * N, ...] for states, actions,
            rewards, next_states, terminals, values, and gae.
        """
        steps, agents = self.n_steps, self.agents.shape[0]
        return TrajectoryDataPPO(
            states=self.states.reshape((steps * agents, *self.states.shape[2:])),
            actions=self.actions.reshape((steps * agents, *self.actions.shape[2:])),
            rewards=self.rewards.reshape((steps * agents,)),
            next_states=self.next_states.reshape(
                (steps * agents, *self.next_states.shape[2:])
            ),
            terminals=self.terminals.reshape((steps * agents,)),
            n_steps=steps * agents,
            agents=jnp.array(1),
            values=self.values.reshape((steps * agents,)),
            gae=self.gae.reshape((steps * agents,)),
            gamma=self.gamma,
            lam=self.lam,
        )  # type: ignore[call-arg]
