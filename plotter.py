import sys
import os
import matplotlib.pyplot as plt
import pandas as pd

experiment_path = sys.argv[1]

print(experiment_path)

results_path_player1 = \
    os.path.join(experiment_path, "player1/progress.csv")
results_path_player0 = \
    os.path.join(experiment_path, "player0/progress.csv")

assert(os.path.exists(results_path_player0))
assert(os.path.exists(results_path_player1))


def parse(path):
    df = pd.read_csv(path, index_col="episodes_total")
    episodes_per_step = df.index.to_list()
    episode_reward_max = df['episode_reward_max'].to_list()
    episode_reward_min = df['episode_reward_min'].to_list()
    episode_reward_mean = df['episode_reward_mean'].to_list()
    assert(len(episode_reward_mean) == len(episode_reward_min) == len(episode_reward_max))
    return (episodes_per_step,
            episode_reward_max,
            episode_reward_min,
            episode_reward_mean)


player1 = parse(results_path_player0)
player2 = parse(results_path_player1)

fig = plt.figure()

plt.subplot(2, 1, 1)
plt.title("player0")
plt.plot(player1[0], player1[3],  label="avg_reward")
plt.plot(player1[0], player1[2],  label="min_reward")
plt.plot(player1[0], player1[1],  label="max_reward")
plt.legend(loc=1)

plt.subplot(2, 1, 2)
plt.title("player1")
plt.plot(player2[0], player2[3],  label="avg_reward")
plt.plot(player2[0], player2[2],  label="min_reward")
plt.plot(player2[0], player2[1],  label="max_reward")
plt.legend(loc=1)


plt.show(block=True)
