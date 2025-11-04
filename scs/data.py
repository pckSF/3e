from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

from flax import (
    struct,
)

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

    def get_batch_data(self, batch_indices: jax.Array) -> TrajectoryData:
        """Returns a batch of data based on the provided indices."""
        return TrajectoryData(
            states=self.states[batch_indices],
            actions=self.actions[batch_indices],
            action_log_densities=self.action_log_densities[batch_indices],
            rewards=self.rewards[batch_indices],
            next_states=self.next_states[batch_indices],
            terminals=self.terminals[batch_indices],
            n_steps=batch_indices.shape[0],
            agents=self.agents,
            samples=True,
        )

    def stack_agent_trajectories(self) -> TrajectoryData:
        """Stacks the agent dimension into the batch dimension.

        Returns:
            A TrajectoryData object with shape [T * N, ...] for states, actions,
            action_log_densities, rewards, next_states, and terminals.
        """
        steps, agents = self.n_steps, self.agents
        return TrajectoryData(
            states=self.states.reshape((steps * agents, *self.states.shape[2:])),
            actions=self.actions.reshape((steps * agents, *self.actions.shape[2:])),
            action_log_densities=self.action_log_densities.reshape(
                (steps * agents, *self.action_log_densities.shape[2:])
            ),
            rewards=self.rewards.reshape((steps * agents,)),
            next_states=self.next_states.reshape(
                (steps * agents, *self.next_states.shape[2:])
            ),
            terminals=self.terminals.reshape((steps * agents,)),
            n_steps=steps * agents,
            agents=1,
            samples=self.samples,
        )


@struct.dataclass
class ValueAndGAE:
    """JAX-friendly PyTree data container that value predictions and the GAE.

    Fields that are scanned over axis 0:
    - values:           [T, N]
    - gae:              [T, N]

    Static metadata:
    - n_steps:         int
    - agents:          int
    - gamma:           float
    - lam:             float
    - samples:         bool
    """

    values: jax.Array
    gae: jax.Array
    n_steps: int = struct.field(pytree_node=False)
    agents: int = struct.field(pytree_node=False)
    gamma: float = struct.field(pytree_node=False)
    lam: float = struct.field(pytree_node=False)
    samples: bool = struct.field(pytree_node=False, default=False)

    @classmethod
    def load(cls, path: str) -> ValueAndGAE:
        """Loads TrajectoryData from a pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(**data)

    def save(self, path: str) -> None:
        """Saves ValueAndGAE to a pickle file."""
        data_dict = {
            "values": self.values,
            "gae": self.gae,
            "n_steps": self.n_steps,
            "agents": self.agents,
            "gamma": self.gamma,
            "lam": self.lam,
            "samples": self.samples,
        }
        with open(path, "wb") as f:
            pickle.dump(data_dict, f)

    @classmethod
    def create_from_trajectory(
        cls,
        trajectory: TrajectoryData,
        model: ActorCritic,
        config: PPOConfig,
    ) -> ValueAndGAE:
        """Creates a ValueAndGAE object from a standard trajectory.

        This method computes value estimates and GAE advantages based on the
        passed trajectory data.

        Must be called on not-sampled Trajectories only!

        Args:
            trajectory: The base trajectory data.
            model: The actor-critic model to compute values.
            config: The PPO configuration containing gamma and lambda.

        Returns:
            A new TrajectoryDataPPO object with PPO-specific data.
        """
        if trajectory.samples:
            raise ValueError(
                "Cannot create ValueAndGAE from sampled TrajectoryData; "
                "GAE computation requires timestep order to be preserved."
            )
        values = model.get_values(trajectory.states)
        next_values = model.get_values(trajectory.next_states)
        # TODO: Normalize Advantages?
        gae = calculate_gae(
            rewards=trajectory.rewards,
            values=values,
            next_values=next_values,
            terminals=trajectory.terminals,
            gamma=config.discount_factor,
            lmbda=config.gae_lambda,
        )
        return cls(
            values=values,
            gae=gae,
            n_steps=trajectory.n_steps,
            agents=trajectory.agents,
            gamma=config.discount_factor,
            lam=config.gae_lambda,
        )

    def get_batch_data(self, batch_indices: jax.Array) -> ValueAndGAE:
        """Returns a batch of data based on the provided indices."""
        return ValueAndGAE(
            values=self.values[batch_indices],
            gae=self.gae[batch_indices],
            n_steps=self.n_steps,
            agents=self.agents,
            gamma=self.gamma,
            lam=self.lam,
            samples=True,
        )

    def stack_agent_trajectories(self) -> ValueAndGAE:
        """Stacks the agent dimension into the batch dimension.

        Returns:
            A ValueAndGAE object with shape [T * N, ...] for values and gae.
        """
        steps, agents = self.n_steps, self.agents
        return ValueAndGAE(
            values=self.values.reshape((steps * agents,)),
            gae=self.gae.reshape((steps * agents,)),
            n_steps=steps * agents,
            agents=1,
            gamma=self.gamma,
            lam=self.lam,
            samples=self.samples,
        )
