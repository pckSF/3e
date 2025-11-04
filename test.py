from __future__ import annotations

from flax import nnx
import gymnasium as gym
import optax

from scs.data_logging import DataLogger
from scs.nn_modules import NNTrainingState
from scs.ppo import train_agent
from scs.ppo.defaults import (
    get_config,
)
from scs.ppo.models import ActorCritic

############################################################################
# Hyperparameters
############################################################################
agent_config = get_config(
    learning_rate=2.5e-4,
    learning_rate_decay=0.9995,
    discount_factor=0.99,
    clip_parameter=0.2,
    entropy_coefficient=0.01,
    gae_lambda=0.95,
    n_actors=8,
    n_actor_steps=128,
    batch_size=64,
    num_epochs=10,
    action_noise=0.0,
    normalize_advantages=True,
)
max_training_loops: int = 10000
seed: int = 0
############################################################################
# Setup logging
logger = DataLogger(log_dir="data/logs/ppo_hopper")

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
lr_decay_schedule = optax.exponential_decay(
    init_value=agent_config.learning_rate,
    transition_steps=agent_config.num_epochs * max_training_loops,
    decay_rate=agent_config.learning_rate_decay,
)
train_state = NNTrainingState.create(
    model_def=nnx.graphdef(model),
    model_state=nnx.state(model, nnx.Param),
    optimizer=optax.adam(lr_decay_schedule),
)

train_state, envs, losses, rewards = train_agent(
    train_state=train_state,
    envs=envs,
    config=agent_config,
    data_logger=logger,
    max_training_loops=max_training_loops,
    rngs=rngs,
)
