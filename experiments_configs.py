from configs import get_player_trainers
from env.gridworld import FOOD_TINY, FOOD_LITTLE, FOOD_NORMAL, FOOD_ALOT


def experiment_scarse(murder_mode=True, steps_per_episode=1000):
    return {
        "size": 42,
        "num_players": 2,
        "murder_mode": murder_mode,
        "steps_per_episode": steps_per_episode,
        "player_move_cost": 0,
        "player_respawn_time": 300,
        "food_reward": 1,
        "food_respawn_time": 100,
        "food_level": FOOD_LITTLE,
        "beam_color_diff": True,
        "draw_beam": True,
        "draw_shooting_direction": True
    }


def experiment_very_scarse(murder_mode=True, steps_per_episode=1000):
    return {
            "size": 42,
            "num_players": 2,
            "murder_mode": murder_mode,
            "steps_per_episode": steps_per_episode,
            "player_move_cost": 0,
            "player_respawn_time": 300,
            "food_reward": 1,
            "food_respawn_time": 20,
            "food_level": FOOD_TINY,
            "beam_color_diff": True,
            "draw_beam": True,
            "draw_shooting_direction": True
    }



# def __abudent(murder_mode, steps_per_episode):
#     return {
#             "size": 42,
#             "num_players": 2,
#             "murder_mode": murder_mode,
#             "steps_per_episode": steps_per_episode,
#             "player_move_cost": 0,
#             "player_respawn_time": 5,
#             "food_reward": 1,
#             "food_respawn_time": 30,
#             "food_level": FOOD_ALOT
#     }


# def __scarse(murder_mode, steps_per_episode):
#     return {
#             "size": 42,
#             "num_players": 2,
#             "murder_mode": murder_mode,
#             "steps_per_episode": steps_per_episode,
#             "player_move_cost": 0,
#             "player_respawn_time": 5,
#             "food_reward": 1,
#             "food_respawn_time": 20,
#             "food_level": FOOD_LITTLE
#     }


# def __very_scarse(murder_mode, steps_per_episode):
#     return {
#             "size": 42,
#             "num_players": 2,
#             "murder_mode": murder_mode,
#             "steps_per_episode": steps_per_episode,
#             "player_move_cost": 0,
#             "player_respawn_time": 5,
#             "food_reward": 1,
#             "food_respawn_time": 10,
#             "food_level": FOOD_LITTLE
#     }

def experiment_DQN_DQN(envq, env_string):
    experiment_configs = {
        "env_configs": envq,
        "agent0_alg": "DQN",
        "agent0_configs": {
            # Discount Factor
            "gamma": 0.95,
            "n_step": 3,
            # size of the replay buffer
            "buffer_size": 50_000,
            # Whether to use dueling dqn
            "dueling": True,
            # Whether to use double dqn
            "double_q": True,
            "explore": True,
            "exploration_config": {
               "type": "EpsilonGreedy",
               "initial_epsilon": 1.0,
               "final_epsilon": 0.02,
               "epsilon_timesteps": 400_000,
            }
        },
        "agent1_alg": "DQN",
        "agent1_configs": {
            # Discount Factor
            "gamma": 0.95,
            "n_step": 3,
            # size of the replay buffer
            "buffer_size": 50_000,
            # Whether to use dueling dqn
            "dueling": True,
            # Whether to use double dqn
            "double_q": True,
            # === EXPLORATION CONFIG" ===
            "explore": True,
            "exploration_config": {
               "type": "EpsilonGreedy",
               "initial_epsilon": 1.0,
               "final_epsilon": 0.02,
               "epsilon_timesteps": 400_000,
            }
        },
        "rid": ("DQN_vs_DQN_" + env_string)
    }
    return experiment_configs


def experiment_DQN_PPO_no_murder(envq, env_string):
    experiment_configs = {
        "env_configs": envq,
        "agent0_alg": "DQN",
        "agent0_configs": {
            # Discount Factor
            "gamma": 0.95,
            "n_step": 3,
            # size of the replay buffer
            "buffer_size": 50_000,
            # Whether to use dueling dqn
            "dueling": True,
            # Whether to use double dqn
            "double_q": True,
            # === EXPLORATION CONFIG" ===
            "explore": True,
            "exploration_config": {
               "type": "EpsilonGreedy",
               "initial_epsilon": 1.0,
               "final_epsilon": 0.02,
               "epsilon_timesteps": 150_000,
            }
        },
        "agent1_alg": "PPO",
        "agent1_configs": {
            "explore": True,
        },
        "rid": ("HIGH_EPSILON_DQN_vs_PPO_no_murder_" + env_string)
    }
    return experiment_configs
