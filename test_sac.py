from __future__ import annotations

from flax import nnx
import gymnasium as gym

from scs.data_logging import DataLogger
from scs.nn_modules import (
    NNTrainingState,
    get_optimizer,
)
from scs.ppo import train_agent
from scs.ppo.defaults import (
    get_config,
)
from scs.ppo.models import PolicyValue

############################################################################
# Hyperparameters
############################################################################
agent_config = get_config(
    learning_rate=1.5e-4,
    learning_rate_end_value=0.0,
    learning_rate_decay=0.9995,
    optimizer="adam",
    lr_schedule="linear",
    discount_factor=0.99,
    clip_parameter=0.2,
    entropy_coefficient=0.01,
    gae_lambda=0.95,
    n_actors=20,
    n_actor_steps=256,
    batch_size=512,
    num_epochs=10,
    value_loss_coefficient=0.2,
    evaluation_frequency=25,
    normalize_advantages=True,
    max_training_loops=10000,
)
seed: int = 0
############################################################################
# Setup logging
logger = DataLogger(log_dir="logs/ppo_hopper")

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
model = PolicyValue(rngs=rngs)
train_state = NNTrainingState.create(
    model_def=nnx.graphdef(model),
    model_state=nnx.state(model, nnx.Param),
    optimizer=get_optimizer(agent_config),
)

train_state, envs, losses, rewards = train_agent(
    train_state=train_state,
    envs=envs,
    config=agent_config,
    data_logger=logger,
    max_training_loops=agent_config.max_training_loops,
    rngs=rngs,
)
