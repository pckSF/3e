from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

from flax import struct
import jax.numpy as jnp

if TYPE_CHECKING:
    import jax


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
