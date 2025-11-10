from __future__ import annotations

from flax import nnx
import gymnasium as gym

from scs.data_logging import DataLogger
from scs.nn_modules import (
    NNTrainingState,
    NNTrainingStateSoftTarget,
    get_optimizer,
)
from scs.sac import train_agent
from scs.sac.defaults import (
    get_config,
)
from scs.sac.models import (
    Policy,
    QValue,
)

############################################################################
# Hyperparameters
############################################################################
agent_config = get_config(
    env_name="Hopper-v5",
    lr_policy=3e-4,
    lr_schedule_policy="linear",
    lr_end_value_policy=0.0,
    lr_decay_policy=0.99,
    optimizer_policy="adam",
    lr_qvalue=3e-4,
    lr_schedule_qvalue="linear",
    lr_end_value_qvalue=0.0,
    lr_decay_qvalue=0.99,
    optimizer_qvalue="adam",
    discount_factor=0.99,
    entropy_coefficient=0.2,
    n_actors=10,
    n_actor_steps=128,
    batch_size=256,
    num_epochs=3,
    save_checkpoints=500,
    evaluation_frequency=25,
    max_training_loops=1000000,
    replay_buffer_size=10000,
    target_network_update_weight=0.005,
)
seed: int = 0
############################################################################
# Setup logging
logger = DataLogger(log_dir="logs/sac_hopper")

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

# Create the models
model_policy = Policy(rngs=rngs)
train_state_policy = NNTrainingState.create(
    model_def=nnx.graphdef(model_policy),
    model_state=nnx.state(model_policy, nnx.Param),
    optimizer=get_optimizer(agent_config, model_policy),
)
model_q1 = QValue(rngs=rngs)
train_state_q1 = NNTrainingStateSoftTarget.create(
    model_def=nnx.graphdef(model_q1),
    model_state=nnx.state(model_q1, nnx.Param),
    optimizer=get_optimizer(agent_config, model_q1),
    tau=agent_config.target_network_update_weight,
)
model_q2 = QValue(rngs=rngs)
train_state_q2 = NNTrainingStateSoftTarget.create(
    model_def=nnx.graphdef(model_q2),
    model_state=nnx.state(model_q2, nnx.Param),
    optimizer=get_optimizer(agent_config, model_q2),
    tau=agent_config.target_network_update_weight,
)

(
    (train_state_policy, train_state_q1, train_state_q2),
    envs,
    losses_policy,
    losses_q,
    rewards,
) = train_agent(
    train_state_policy=train_state_policy,
    train_state_q1=train_state_q1,
    train_state_q2=train_state_q2,
    envs=envs,
    config=agent_config,
    data_logger=logger,
    max_training_loops=agent_config.max_training_loops,
    rngs=rngs,
)
