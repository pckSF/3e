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
from scs.ppo.models import (
    PolicyValue,
)

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

# Define which parameters belong to policy vs value
policy_param_names = {
    "policy_linear_1",
    "policy_layernorm_1",
    "policy_linear_2",
    "policy_layernorm_2",
    "policy_mean",
    "policy_log_std",
}


# Label function for parameter partitioning
# optax.multi_transform expects a function that takes the params pytree
# and returns a pytree with the same structure but with labels instead of values
def param_labels(params):
    import jax

    def label_fn(path, _):
        # path is a tuple of keys leading to this parameter
        # Check if any of the path components match policy parameter names
        for key in path:
            if key in policy_param_names:
                return "policy"
        return "value"

    return jax.tree_util.tree_map_with_path(label_fn, params)


# Create multi-transform optimizer with different learning rates
# ppo_lr_adam: 1.7e-4 for policy
# val_lr: 4e-4 for value
optimizer = optax.multi_transform(
    transforms={
        "policy": optax.adam(learning_rate=1.7e-4),
        "value": optax.adam(learning_rate=4e-4),
    },
    param_labels=param_labels,
)

train_state = NNTrainingState.create(
    model_def=nnx.graphdef(model),
    model_state=nnx.state(model, nnx.Param),
    optimizer=optimizer,
)

train_state, envs, losses, rewards = train_agent(
    train_state=train_state,
    envs=envs,
    config=agent_config,
    data_logger=logger,
    max_training_loops=agent_config.max_training_loops,
    rngs=rngs,
)
