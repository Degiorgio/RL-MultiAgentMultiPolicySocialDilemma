from env.gridworld import GatheringEnv

def env_factory(env_config):
    if "randomized_food" not in env_config:
        env_config["randomized_food"] = False
    env = GatheringEnv(**env_config)
    return env
