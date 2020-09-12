from env.configs import env_scarse, env_very_scarse, env_abudent, env_normal

def expr_environment():
    exp1 = experiment_DQN_PPO(env_scarse(player_respawn_time=200, food_respawn_time=50),
                              tt=4,
                              env_string="scarse_high_n_f")
    exp2 = experiment_DQN_PPO(env_abudent(player_respawn_time=200, food_respawn_time=5),
                              tt=4,
                              env_string="abundant_high_n_p")
    exp3 = experiment_DQN_PPO(env_abudent(player_respawn_time=2, food_respawn_time=5),
                              tt=4,
                              env_string="abundant_low_n_p")
    exp4 = experiment_DQN_PPO(env_scarse(player_respawn_time=200, food_respawn_time=14),
                              tt=4,
                              env_string="scarse_high_n_p")
    exp5 = experiment_DQN_PPO(env_scarse(player_respawn_time=2, food_respawn_time=14),
                              tt=4,
                              env_string="scarse_low_n_f")
    return [exp1, exp2] #exp3, exp4, exp5]


def expr_parameters():
    env = env_scarse(player_respawn_time=50, food_respawn_time=14)
    exp1 = experiment_DQN_PPO(env,
                              env_string="DQN_LOW_REPLAY_BUFF",
                              dqn_buffer_size=500)
    exp2 = experiment_DQN_PPO(env,
                              env_string="DQN_LOW_DISCOUMT",
                              dqn_gamma=0.6)
    exp3 = experiment_DQN_PPO(env,
                              env_string="PPO_LOW_CLIPPING",
                              ppo_clipping=0.1)
    exp4 = experiment_DQN_PPO(env,
                              env_string="PPO_NO_CRITIC",
                              ppo_critic=False,
                              ppo_gae=False,
                              ppo_batch_mode="complete_episodes")
    return [exp1, exp2, exp3, exp4]


def expr_exploration_vs_exploitation():
    exploration_config_low = {
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.0,
            "final_epsilon": 0.02,
            "epsilon_timesteps": 80_000,
    }
    exploration_config_high = {
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.0,
            "final_epsilon": 0.06,
            "epsilon_timesteps": 600_000,
    }
    env = env_scarse(player_respawn_time=50, food_respawn_time=14)
    exp1 = experiment_DQN_PPO(env,
                              exploration_config_dqn=exploration_config_low,
                              exploration_config_PPO=exploration_config_low,
                              env_string="scarse_low_vs_low")
    exp2 = experiment_DQN_PPO(env,
                              exploration_config_dqn=exploration_config_high,
                              exploration_config_PPO=exploration_config_high,
                              env_string="scarse_high_vs_high")
    exp3 = experiment_DQN_PPO(env,
                              exploration_config_dqn=exploration_config_high,
                              exploration_config_PPO=exploration_config_low,
                              env_string="scarse_dqn_high_vs_ppo_low")
    exp4 = experiment_DQN_PPO(env,
                              exploration_config_dqn=exploration_config_low,
                              exploration_config_PPO=exploration_config_high,
                              env_string="scarse_dqn_low_vs_ppo_high")
    return [exp1, exp2, exp3, exp4]


def experiment_DQN_DQN(envq, env_string, tt=4,
                       exploration_config_dqn=None,
                       dqn_buffer_size_1=50_000,
                       dqn_buffer_size_2=50_000,
                       dqn_gamma=0.95,
                       dqn_batch_size=32):
    if exploration_config_dqn is None:
        exploration_config_dqn = {
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.0,
            "final_epsilon": 0.02,
            "epsilon_timesteps": 250_000,
        }
    experiment_configs = {
        "env_configs": envq,
        "agent0_alg": "DQN",
        "agent0_configs": {
            # Discount Factor
            "gamma": dqn_gamma,
            "n_step": 3,
            # size of the replay buffer
            "buffer_size": dqn_buffer_size_1,
            # Whether to use dueling dqn
            "dueling": True,
            # Whether to use double dqn
            "double_q": True,
            # === EXPLORATION CONFIG" ===
            "explore": True,
            "exploration_config": exploration_config_dqn,
            "timesteps_per_iteration": 1000*tt,
            "train_batch_size": dqn_batch_size,
        },
        "agent1_alg": "DQN",
        "agent1_configs": {
            # Discount Factor
            "gamma": dqn_gamma,
            "n_step": 3,
            # size of the replay buffer
            "buffer_size": dqn_buffer_size_2,
            # Whether to use dueling dqn
            "dueling": True,
            # Whether to use double dqn
            "double_q": True,
            # === EXPLORATION CONFIG" ===
            "explore": True,
            "exploration_config": exploration_config_dqn,
            "timesteps_per_iteration": 1000*tt,
            "train_batch_size": dqn_batch_size,
        },
        "rid": ("DQN_vs_DQN" + env_string)
    }
    return experiment_configs


def experiment_PPO_PPO(envq, env_string, tt=4,
                       ppo_batch_size=128,
                       ppo_clipping_1=0.3,
                       ppo_clipping_2=0.3,
                       ppo_critic=True,
                       ppo_batch_mode="truncate_episodes",
                       ppo_gae=True):

    AGENT1_CONFIG = {
        "explore": True,
        "use_critic": ppo_critic,
        "batch_mode": ppo_batch_mode,
        "use_gae": ppo_gae,
        "sgd_minibatch_size": ppo_batch_size,
        "clip_param": ppo_clipping_1,
        "train_batch_size": 1000*tt,
    }
    AGENT2_CONFIG = {
        "explore": True,
        "use_critic": ppo_critic,
        "batch_mode": ppo_batch_mode,
        "use_gae": ppo_gae,
        "sgd_minibatch_size": ppo_batch_size,
        "clip_param": ppo_clipping_2,
        "train_batch_size": 1000*tt,
    }
    experiment_configs = {
        "env_configs": envq,
        "agent0_alg": "PPO",
        "agent0_configs": AGENT1_CONFIG,
        "agent1_alg": "PPO",
        "agent1_configs": AGENT2_CONFIG,
        "rid": ("PPO_vs_PPO" + env_string)
    }
    return experiment_configs


def experiment_DQN_PPO(envq, env_string, tt=4,
                       exploration_config_dqn=None,
                       exploration_config_PPO=None,
                       dqn_buffer_size=50_000,
                       dqn_gamma=0.95,
                       dqn_batch_size=32,
                       ppo_batch_size=128,
                       ppo_clipping=0.3,
                       ppo_critic=True,
                       ppo_batch_mode="truncate_episodes",
                       ppo_gae=True):

    if exploration_config_dqn is None:
        exploration_config_dqn = {
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.0,
            "final_epsilon": 0.02,
            "epsilon_timesteps": 250_000,
        }

    if exploration_config_PPO is None:
        AGENT1_CONFIG = {
            "explore": True,
            "use_critic": ppo_critic,
            "batch_mode": ppo_batch_mode,
            "use_gae": ppo_gae,
            "sgd_minibatch_size": ppo_batch_size,
            "clip_param": ppo_clipping,
            "train_batch_size": 1000*tt,
        }
    else:
        AGENT1_CONFIG = {
            "explore": True,
            "use_critic": ppo_critic,
            "use_gae": ppo_gae,
            "batch_mode": ppo_batch_mode,
            "clip_param": ppo_clipping,
            "sgd_minibatch_size": ppo_batch_size,
            "exploration_config": exploration_config_PPO,
            "train_batch_size": 1000*tt,
        }

    experiment_configs = {
        "env_configs": envq,
        "agent0_alg": "DQN",
        "agent0_configs": {
            # Discount Factor
            "gamma": dqn_gamma,
            "n_step": 3,
            # size of the replay buffer
            "buffer_size": dqn_buffer_size,
            # Whether to use dueling dqn
            "dueling": True,
            # Whether to use double dqn
            "double_q": True,
            # === EXPLORATION CONFIG" ===
            "explore": True,
            "exploration_config": exploration_config_dqn,
            "timesteps_per_iteration": 1000*tt,
            "train_batch_size": dqn_batch_size,
        },
        "agent1_alg": "PPO",
        "agent1_configs": AGENT1_CONFIG,
        "rid": ("DQN_vs_PPO" + env_string)
    }
    return experiment_configs
