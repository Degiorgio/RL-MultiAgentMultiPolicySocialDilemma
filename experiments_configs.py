from configs import get_player_trainers
from env.gridworld import FOOD_TINY, FOOD_LITTLE, FOOD_NORMAL, FOOD_ALOT


def experiment_DQN_DQN_no_murder():
    experiment_configs = {
        "env_configs": {
            "size": 42,
            "num_players": 2,
            "murder_mode": False,
            "steps_per_episode": 200,
            "player_move_cost": 1,
            "player_respawn_time": 5,
            "food_reward": 25,
            "food_respawn_time": 40,
            "food_level": FOOD_NORMAL
        },
        "agent0_alg": "DQN",
        "agent0_configs": {
            # Discount Factor
            "gamma": 0.95,
            "n_step": 3,
            # size of the replay buffer
            "buffer_size": 50_000,
            # Whether to use dueling dqn
            "dueling": False,
            # Whether to use double dqn
            "double_q": False,
            # === EXPLORATION CONFIG" ===
            "explore": True,
            "exploration_config": {
               "type": "EpsilonGreedy",
               "initial_epsilon": 1.0,
               "final_epsilon": 0.02,
               "epsilon_timesteps": 10000,
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
               "epsilon_timesteps": 10000,
            }
        },
        "rid": "DQN_vs_DQN_no_murder"
    }
    return experiment_configs
