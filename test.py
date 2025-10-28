from __future__ import annotations

import gymnasium as gym

############################################################################
# Hyperparameters
############################################################################
learning_rate_online: float = 0.0002
gamma: float = 0.99
batchsize: int = 32
max_training_episodes: int = 2500
max_steps: int = 250
reward_threshold: int = 100
n_actors: int = 4
seed: int = 0
############################################################################

env = gym.make("Hopper-v5")
env.reset(seed=seed)

# Setup vectorized evaluation environment
envs = gym.vector.SyncVectorEnv(
    [lambda: gym.make("Hopper-v5") for _ in range(n_actors)],
    autoreset_mode=gym.vector.AutoresetMode.DISABLED,
)
envs.reset(seed=seed)
