from env.gridworld import FOOD_TINY, FOOD_LITTLE, FOOD_NORMAL, FOOD_ALOT

SIZE = 42


    # exp3 = experiment_DQN_DQN(
    #     env_scarse(small_world=False),  "scarse_big_world")
    # exp4 = experiment_DQN_DQN(
    #     env_scarse(player_move_cost=0),  "scarse_no_move_penality")
def expr_scrase():
    exp1 = experiment_DQN_PPO(env_scarse(player_move_cost=0), env_string="scarse_nomove")
    exp4 = experiment_DQN_DQN(env_scarse(player_move_cost=0, shoot_in_all_directions=False, draw_shooting_direction=False), "scarse_shoot_one_direction_noaid")
    exp2 = experiment_DQN_DQN(env_scarse(player_move_cost=0, shoot_in_all_directions=False), "scarse_shoot_one_direction")
    exp5 = experiment_DQN_DQN(env_abudent(player_move_cost=0), "abudent_no_move_penality")
    exp3 = experiment_DQN_DQN(env_normal(player_move_cost=0), "normal_no_move_penality")
    return [exp1, exp2, exp3, exp4, exp5]


def env_scarse(murder_mode=True,
               steps_per_episode=1000,
               beam_color_diff=True,
               draw_beam=True,
               draw_shooting_direction=True,
               shoot_in_all_directions=True,
               small_world=True,
               player_move_cost=1):
    return {
        "size": SIZE,
        "num_players": 2,
        "murder_mode": murder_mode,
        "steps_per_episode": steps_per_episode,
        "player_move_cost": player_move_cost,
        "player_respawn_time": 200,
        "food_reward": 50,
        "food_respawn_time": 14,
        "food_level": FOOD_LITTLE,
        "beam_color_diff": beam_color_diff,
        "draw_beam": draw_beam,
        "draw_shooting_direction": draw_shooting_direction,
        "shoot_in_all_directions": shoot_in_all_directions,
        "small_world":  small_world
    }


def env_very_scarse(murder_mode=True, steps_per_episode=1000, beam_color_diff=True, draw_beam=True, draw_shooting_direction=True, shoot_in_all_directions=True, small_world=True):
    return {
            "size": SIZE,
            "num_players": 2,
            "murder_mode": murder_mode,
            "steps_per_episode": steps_per_episode,
            "player_move_cost": 1,
            "player_respawn_time": 200,
            "food_reward": 50,
            "food_respawn_time": 5,
            "food_level": FOOD_TINY,
            "beam_color_diff": beam_color_diff,
            "draw_beam": draw_beam,
            "draw_shooting_direction": draw_shooting_direction,
            "shoot_in_all_directions": shoot_in_all_directions,
            "small_world":  small_world
    }


def env_abudent(murder_mode=True, steps_per_episode=1000, beam_color_diff=True, draw_beam=True, draw_shooting_direction=True, shoot_in_all_directions=True, small_world=True, player_move_cost=1):
    return {
            "size": SIZE,
            "num_players": 2,
            "murder_mode": murder_mode,
            "steps_per_episode": steps_per_episode,
            "player_move_cost": player_move_cost,
            "player_respawn_time": 200,
            "food_reward": 50,
            "food_respawn_time": 110,
            "food_level": FOOD_ALOT,
            "beam_color_diff": beam_color_diff,
            "draw_beam": draw_beam,
            "draw_shooting_direction": draw_shooting_direction,
            "shoot_in_all_directions": shoot_in_all_directions,
            "small_world":  small_world
    }


def env_normal(murder_mode=True,
               steps_per_episode=1000,
               beam_color_diff=True,
               draw_beam=True,
               draw_shooting_direction=True,
               shoot_in_all_directions=True,
               small_world=True,
               player_move_cost=1):
    return {
            "size": SIZE,
            "num_players": 2,
            "murder_mode": murder_mode,
            "steps_per_episode": steps_per_episode,
            "player_move_cost": player_move_cost,
            "player_respawn_time": 200,
            "food_reward": 50,
            "food_respawn_time": 40,
            "food_level": FOOD_NORMAL,
            "beam_color_diff": beam_color_diff,
            "draw_beam": draw_beam,
            "draw_shooting_direction": draw_shooting_direction,
            "shoot_in_all_directions": shoot_in_all_directions,
            "small_world":  small_world
    }


def experiment_DQN_DQN(envq, env_string, pl_1=100_000, pl_2=200_000):
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
               "epsilon_timesteps": pl_1,
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
               "epsilon_timesteps": pl_2,
            }
        },
        "rid": ("DQN_vs_DQN_" + env_string)
    }
    return experiment_configs


def experiment_DQN_PPO(envq, env_string):
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
               "epsilon_timesteps": 250_000,
            }
        },
        "agent1_alg": "PPO",
        "agent1_configs": {
            "explore": True,
        },
        "rid": ("DQN_vs_PPO" + env_string)

    }
    return experiment_configs
