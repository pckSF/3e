from __future__ import annotations

from flax import nnx
import gymnasium as gym
import jax
import numpy as np
import optax

from scs.data import ValueAndGAE
from scs.nn_modules import NNTrainingState
from scs.ppo.agent import loss_fn
from scs.ppo.defaults import (
    PPOConfig,
    get_config,
)
from scs.ppo.models import ActorCritic
from scs.ppo.rollouts import collect_trajectories

############################################################################
# Hyperparameters
############################################################################
hyperparameters: dict[str, int | float] = {
    "learning_rate": 2.5e-4,
    "discount_factor": 0.99,
    "clip_parameter": 0.1,
    "entropy_coefficient": 0.01,
    "gae_lambda": 0.95,
    "n_actors": 4,
    "n_actor_steps": 128,
    "batch_size": 64,
    "num_epochs": 5,
    "action_noise": 0.2,
}
seed: int = 0
############################################################################

# Create agent configuration
agent_config: PPOConfig = get_config(**hyperparameters)

# Setup RNG
rngs = nnx.Rngs(
    seed,
    config=seed + 1,
    action_select=seed + 2,
    noise=seed + 3,
    sample=seed + 4,
    trajectory=seed + 5,
)
# Setup vectorized environment
envs = gym.vector.SyncVectorEnv(
    [lambda: gym.make("Hopper-v5") for _ in range(agent_config.n_actors)],
    autoreset_mode=gym.vector.AutoresetMode.DISABLED,
)
envs.reset(seed=seed)

# Create the model
model = ActorCritic(rngs=rngs)
train_state = NNTrainingState.create(
    model_def=nnx.graphdef(model),
    model_state=nnx.state(model, nnx.Param),
    optimizer=optax.adam(agent_config.learning_rate),
)

reset_mask = np.ones((agent_config.n_actors,), dtype=bool)
state: np.ndarray = envs.reset()[0]
trajectory, reset, state = collect_trajectories(
    model=model,
    envs=envs,
    reset_mask=reset_mask,
    state=state,
    rng=rngs,
    config=agent_config,
)

value_and_gae = ValueAndGAE.create_from_trajectory(
    trajectory=trajectory,
    model=model,
    config=agent_config,
)

trajectory_flat = trajectory.stack_agent_trajectories()
value_and_gae_flat = value_and_gae.stack_agent_trajectories()

batch_indices = jax.random.choice(
    rngs.sample(),
    a=trajectory_flat.n_steps,
    shape=(agent_config.batch_size,),
    replace=False,
)

trajectory_batch = trajectory_flat.get_batch_data(batch_indices)
value_and_gae_batch = value_and_gae_flat.get_batch_data(batch_indices)

loss = loss_fn(
    model=model,
    batch=trajectory_batch,
    batch_computations=value_and_gae_batch,
    config=agent_config,
)
