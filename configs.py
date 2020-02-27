from ray.rllib.agents.dqn.apex import ApexTrainer
from ray.rllib.agents.dqn.dqn import DQNTrainer

from ray.rllib.agents.dqn.dqn_policy import DQNTFPolicy


def policy_mapping_fn(agent_id):
    if agent_id == "player0":
        return "0"
    elif agent_id == "player1":
        return "1"
    else:
        raise Exception("invalid id")


def get_agent_policies(murder_mode):
    from env.gridworld import GatheringEnv
    single_env = GatheringEnv(size=42,
                              num_players=2,
                              player_murder_mode=murder_mode)
    obs_space = single_env.observation_space
    act_space = single_env.action_space
    print(single_env.action_space)
    policies = {
        "0": (DQNTFPolicy, obs_space, act_space, {}),
        "1": (DQNTFPolicy, obs_space, act_space, {}),
    }
    return policies, policy_mapping_fn

def get_player_trainers_apex(evaluate, murder_mode):
    policies, policy_mapping = get_agent_policies(murder_mode)
    dqn_trainer1 = ApexTrainer(
        env="gathering",
        config={
            "num_gpus": 0.45,
            "num_workers": 2,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping,
                "policies_to_train": ["0"],
            },
            "monitor": True,
            "gamma": 0.95,
            # size of the replay buffer
            "buffer_size": 1_000,
            # Whether to use dueling dqn
            "dueling": True,
            # Whether to use double dqn
            "double_q": True,
            "sample_batch_size": 20,
        })

    dqn_trainer2 = DQNTrainer(
        env="gathering",
        config={
            "num_gpus": 0.45,
            "num_workers": 1,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping,
                "policies_to_train": ["1"],
            },
            "gamma": 0.95,
            # size of the replay buffer
            "buffer_size": 20_000,
            # Whether to use dueling dqn
            "dueling": True,
            # Whether to use double dqn
            "double_q": True,
            "monitor": True,
        })
    return dqn_trainer1, dqn_trainer2


def get_player_trainers(evaluate, murder_mode):
    policies, policy_mapping = get_agent_policies(murder_mode)
    dqn_trainer1 = DQNTrainer(
        env="gathering",
        config={
            "num_gpus": 0.5,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping,
                "policies_to_train": ["0"],
            },
            # Discount Factor
            "gamma": 0.95,
            "n_step": 3,
            # size of the replay buffer
            "buffer_size": 1_000,
            # Whether to use dueling dqn
            "dueling": True,
            # Whether to use double dqn
            "double_q": True,
            #  == DEBUG ==
            "monitor": True,
        })

    dqn_trainer2 = DQNTrainer(
        env="gathering",
        config={
            "num_gpus": 0.5,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping,
                "policies_to_train": ["1"],
            },
            # Discount Factor
            "gamma": 0.95,
            "n_step": 3,
            # size of the replay buffer
            "buffer_size": 50_000,
            # Whether to use dueling dqn
            "dueling": True,
            # Whether to use double dqn
            "double_q": True,
            #== DEBUG ==
            "monitor": True,
        })
    return dqn_trainer1, dqn_trainer2
