from env import GatheringEnv
import torch

SEED = 1
EPISODES = 1000
DISCOUNT = 0.99

env = GatheringEnv(n_agents=1)
env.seed(1)
torch.manual_seed(1)

print("observation space max values:", env.observation_space.high)
print("observation space low values:", env.observation_space.low)


for episode in range(EPISODES):
    episode_reward = 0
    initial_state = env.reset()
    import ipdb; ipdb.set_trace()
