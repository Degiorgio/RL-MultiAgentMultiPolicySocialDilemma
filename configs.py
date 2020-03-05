import os
import ray
import ray.rllib
from ray.tune.registry import register_env
from ray.tune.logger import UnifiedLogger

from ray.rllib.agents.dqn.dqn import DQNTrainer
from ray.rllib.agents.dqn.dqn_policy import DQNTFPolicy

from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy

from util import create_dir

DEFAULT_RESULTS_DIR = "out"


def _get_tf(pol):
    if pol == "DQN":
        return DQNTFPolicy
    elif pol == "PPO":
        return PPOTFPolicy
    else:
        raise Exception("not supported")


def _get_trainer_constructor(pol):
    if pol == "DQN":
        return DQNTrainer
    elif pol == "PPO":
        return PPOTrainer
    else:
        raise Exception("not supported")


def env_creator(env_config):
    from env.gridworld import GatheringEnv
    if "randomized_food" not in env_config:
        env_config["randomized_food"] = False
    env = GatheringEnv(**env_config)
    return env


def get_player_trainers(evaluate, experiment_configs):
    def policy_mapping_fn(agent_id):
        if agent_id == "player0":
            return "0"
        elif agent_id == "player1":
            return "1"
        else:
            raise Exception("invalid id")

    def get_agent_policies(env_config, agent0_alg, agent1_alg):
        single_env = env_creator(env_config)
        obs_space = single_env.observation_space
        act_space = single_env.action_space
        print(single_env.action_space)
        policies = {
            "0": (_get_tf(agent0_alg), obs_space, act_space, {}),
            "1": (_get_tf(agent1_alg), obs_space, act_space, {}),
        }
        return policies, policy_mapping_fn

    ray.init()
    register_env("gathering", env_creator)

    EXPERIMENT_PATH = os.path.join(DEFAULT_RESULTS_DIR, experiment_configs['rid'])
    if not evaluate:
        create_dir(EXPERIMENT_PATH, clean=True)
        import json
        with open(os.path.join(EXPERIMENT_PATH, "experiment.json"), 'w') as fp:
            json.dump(experiment_configs, fp, indent=4)

    env_config = experiment_configs['env_configs']

    agent0_constructor = _get_trainer_constructor(experiment_configs['agent0_alg'])
    agent1_constructor = _get_trainer_constructor(experiment_configs['agent1_alg'])
    policies, policy_mapping = \
        get_agent_policies(
            env_config,
            experiment_configs['agent0_alg'],
            experiment_configs['agent1_alg']
        )


    agent0_default_config = {
        "env_config": env_config,
        "num_gpus": 0.5,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping,
            "policies_to_train": ["0"]
        }
    }

    agent1_default_config = {
        "env_config": env_config,
        "num_gpus": 0.5,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping,
            "policies_to_train": ["1"]
        }
    }

    agent0_configs = \
        {**agent0_default_config, **experiment_configs['agent0_configs']}
    agent1_configs = \
        {**agent1_default_config, **experiment_configs['agent1_configs']}

    if evaluate:
        logger_creator = None
    else:
        def default_logger_creator(config):
            """Creates a Unified logger with a default logdir prefix
            containing the agent name and the env id
            """
            if config['multiagent']['policies_to_train'][0] == '0':
                agent_path = os.path.join(EXPERIMENT_PATH, "player0")
            elif config['multiagent']['policies_to_train'][0] == '1':
                agent_path = os.path.join(EXPERIMENT_PATH, "player1")
            create_dir(agent_path, clean=True)
            return UnifiedLogger(config, agent_path, loggers=None)

        logger_creator = default_logger_creator

    agent0 = agent0_constructor(
        env="gathering",
        logger_creator=logger_creator,
        config=agent0_configs
    )

    agent1 = agent1_constructor(
        env="gathering",
        logger_creator=logger_creator,
        config=agent1_configs
    )

    return agent0, agent1
