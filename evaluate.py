import ray
import ray.rllib
from ray.tune.registry import register_env
from configs import get_player_trainers
from train import MURDER_MODE

import numpy as np

# model_path_player0 = "/home/kurt/ray_results/DQN_gathering_2020-02-24_20-06-05akl10q_v/checkpoint_1/checkpoint-1"
# model_path_player1 = "/home/kurt/ray_results/DQN_gathering_2020-02-24_20-06-05akl10q_v/checkpoint_1/checkpoint-1"

# model_path_player0="/home/kurt/ray_results/DQN_gathering_2020-02-24_22-59-34b9yzsvsg/checkpoint_301/checkpoint-301"
# model_path_player1="/home/kurt/ray_results/DQN_gathering_2020-02-24_22-59-37hhb5hqbf/checkpoint_301/checkpoint-301"

model_path_player0="/home/kurt/ray_results/DQNvsDQN_Player0_MurderMode/checkpoint_2501/checkpoint-2501"
model_path_player1="/home/kurt/ray_results/DQNvsDQN_Player1_MurderMode/checkpoint_2501/checkpoint-2501"

51

def play_game(trainerplayer0, trainerplayer1, episodes=5):
    from collections import Counter
    env = env_creator(None)
    player0_action_distribution = [0]*env._get_action_space()
    player1_action_distribution = [0]*env._get_action_space()
    average_reward = Counter({"player0": 0, "player1": 0})
    for episode in range(episodes):
        current_state = env.reset()
        cum_rewards = Counter({"player0": 0, "player1": 0})
        while True:
            if trainerplayer1 is None:
                # Use Random Policy
                actions = {
                    "player0": trainerplayer0.compute_action(current_state["player0"], policy_id="0"),
                    "player1": np.random.randint(0, len(player1_action_distribution))
                }
            else:
                actions = {
                    "player0": trainerplayer0.compute_action(current_state["player0"], policy_id="0"),
                    "player1": trainerplayer1.compute_action(current_state["player1"], policy_id="1")
                }
            player0_action_distribution[actions["player0"]] += 1
            player1_action_distribution[actions["player1"]] += 1
            new_observaton, rewards, done, info = env.step(actions)
            cum_rewards.update(rewards)
            current_state = new_observaton
            if done['__all__']:
                break
        print("rewards",  cum_rewards)
        average_reward.update(cum_rewards)
    player0_action_distribution = \
        np.array(player0_action_distribution)/episodes
    player1_action_distribution = \
        np.array(player1_action_distribution)/episodes
    print("action distribution player 0", player0_action_distribution)
    print("action distribution player 1", player1_action_distribution)
    average_reward['player0'] = average_reward['player0']/episodes
    average_reward['player1'] = average_reward['player1']/episodes
    print(average_reward)


def env_creator(env_config):
    from env.gridworld import FOOD_TINY, FOOD_LITTLE, FOOD_NORMAL, FOOD_ALOT
    from env.gridworld import GatheringEnv
    env = GatheringEnv(size=42,
                       num_players=2,
                       max_steps=200,
                       player_move_cost=1,
                       player_respawn_time=5,
                       player_murder_mode=MURDER_MODE,
                       food_reward=25,
                       food_respawn_time=10,
                       food_level=FOOD_ALOT)
    return env


if __name__ == "__main__":
    ray.init()
    register_env("gathering", env_creator)

    trainer0, trainer1 = \
        get_player_trainers(evaluate=True, murder_mode=MURDER_MODE)

    trainer0.restore(model_path_player0)
    trainer1.restore(model_path_player1)

    # play_game(trainer0, None, episodes=5)
    play_game(trainer0, trainer1, episodes=5)
