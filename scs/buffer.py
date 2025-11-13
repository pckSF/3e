from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from scs.data import TrajectoryData


class ReplayBuffer:
    """Circular replay buffer for off-policy RL algorithms.

    Once full, old transitions are overwritten in FIFO order.
    """

    def __init__(self, max_size: int, observation_dim: int, action_dim: int) -> None:
        """Initializes replay buffer with pre-allocated arrays."""
        self.max_size: int = max_size
        self.current_index: int = 0
        self.available_samples: int = 0
        self.full: bool = False
        self.observations: np.ndarray = np.zeros(
            (max_size, observation_dim), dtype=np.float32
        )
        self.actions: np.ndarray = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards: np.ndarray = np.zeros((max_size,), dtype=np.float32)
        self.next_observations: np.ndarray = np.zeros(
            (max_size, observation_dim), dtype=np.float32
        )
        self.terminals: np.ndarray = np.zeros((max_size,), dtype=np.bool_)

    def add_transition(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        terminal: bool,
    ) -> None:
        """Adds a single transition to the buffer."""
        self.observations[self.current_index] = observation
        self.actions[self.current_index] = action
        self.rewards[self.current_index] = reward
        self.next_observations[self.current_index] = next_observation
        self.terminals[self.current_index] = terminal

        self.current_index += 1
        if self.current_index >= self.max_size:
            self.current_index = 0
            self.full = True

        if self.full:
            self.available_samples = self.max_size
        else:
            self.available_samples += 1

    def add_transitions(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        terminals: np.ndarray,
    ) -> None:
        """Adds a sequence of transitions, handling wraparound if necessary.

        When the sequence insert would exceed buffer capacity, it's split into
        two chunks: one filling to the end of the buffer, and the remainder
        being inserted at the beginning of the buffer arryays.

        Args:
            observations: Batch of observations (batch_size, observation_dim).
            actions: Batch of actions (batch_size, action_dim).
            rewards: Batch of scalar rewards (batch_size,).
            next_observations: Batch of next observations (batch_size, observation_dim).
            terminals: Batch of episode termination flags (batch_size,).
        """
        n = observations.shape[0]
        if n > self.max_size:
            raise ValueError(
                f"Number of transitions to add exceeds buffer maximum size; "
                f"got {n} transitions, but buffer max size is {self.max_size}."
            )

        end_idx = self.current_index + n

        if end_idx > self.max_size:
            # Insert first chunk from current index to end of buffer
            first_chunk = self.max_size - self.current_index
            self.observations[self.current_index : self.max_size] = observations[
                :first_chunk
            ]
            self.actions[self.current_index : self.max_size] = actions[:first_chunk]
            self.rewards[self.current_index : self.max_size] = rewards[:first_chunk]
            self.next_observations[self.current_index : self.max_size] = (
                next_observations[:first_chunk]
            )
            self.terminals[self.current_index : self.max_size] = terminals[:first_chunk]

            # Insert remaining transitions at the start of the buffer
            overflow = end_idx - self.max_size
            self.observations[:overflow] = observations[first_chunk:]
            self.actions[:overflow] = actions[first_chunk:]
            self.rewards[:overflow] = rewards[first_chunk:]
            self.next_observations[:overflow] = next_observations[first_chunk:]
            self.terminals[:overflow] = terminals[first_chunk:]

            self.current_index = overflow
            self.full = True
        else:
            # All samples fit without wrapping
            self.observations[self.current_index : end_idx] = observations
            self.actions[self.current_index : end_idx] = actions
            self.rewards[self.current_index : end_idx] = rewards
            self.next_observations[self.current_index : end_idx] = next_observations
            self.terminals[self.current_index : end_idx] = terminals

            self.current_index = end_idx
            if self.current_index >= self.max_size:
                self.current_index = 0
                self.full = True

        if self.full:
            self.available_samples = self.max_size
        else:
            self.available_samples = self.current_index

    def to_trajectory_data(self) -> TrajectoryData:
        """Converts entire buffer contents to TrajectoryData format."""
        return TrajectoryData(
            observations=jnp.asarray(self.observations, dtype=jnp.float32),
            actions=jnp.asarray(self.actions, dtype=jnp.float32),
            action_log_densities=jnp.zeros(
                (self.max_size, self.actions.shape[1]), dtype=jnp.float32
            ),  # Placeholder
            rewards=jnp.asarray(self.rewards, dtype=jnp.float32),
            next_observations=jnp.asarray(self.next_observations, dtype=jnp.float32),
            terminals=jnp.asarray(self.terminals, dtype=jnp.uint32),
            n_steps=self.available_samples,
            agents=1,
            samples=True,
        )

    def sample_batch(self, indices: np.ndarray) -> TrajectoryData:
        """Samples batch or n batches by index.

        Args:
            indices: Array of indices to sample (batch_size,) or (batch_size, n).
        """
        return TrajectoryData(
            observations=jnp.asarray(self.observations[indices], dtype=jnp.float32),
            actions=jnp.asarray(self.actions[indices], dtype=jnp.float32),
            action_log_densities=jnp.zeros(
                (indices.shape[0],), dtype=jnp.float32
            ),  # Placeholder
            rewards=jnp.asarray(self.rewards[indices], dtype=jnp.float32),
            next_observations=jnp.asarray(
                self.next_observations[indices], dtype=jnp.float32
            ),
            terminals=jnp.asarray(self.terminals[indices], dtype=jnp.uint32),
            n_steps=indices.shape[0],
            agents=1,
            samples=True,
        )
