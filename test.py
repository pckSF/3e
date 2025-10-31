from __future__ import annotations

from flax import nnx
import gymnasium as gym
import optax

from scs.nn_modules import NNTrainingState
from scs.ppo.defaults import (
    PPOConfig,
    get_config,
)
from scs.ppo.models import ActorCritic
from scs.ppo.rollouts import evaluation_trajectory

############################################################################
# Hyperparameters
############################################################################
hyperparameters = {
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

rewards = evaluation_trajectory(
    model=model,
    envs=envs,
    rng=rngs,
    config=agent_config,
)
