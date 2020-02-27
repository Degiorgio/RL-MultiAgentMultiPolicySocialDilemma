import ray
import ray.rllib
from tqdm import tqdm
from ray.tune.registry import register_env
from configs import get_player_trainers, get_player_trainers_apex

NUM_EPISODES = 5_0000
CHECKPOIINT_INTERVAL = 50
MURDER_MODE = True
STEPS_PER_EPISODE = 200

def env_creator(env_config):
    from env.gridworld import FOOD_TINY, FOOD_LITTLE, FOOD_NORMAL, FOOD_ALOT
    from env.gridworld import GatheringEnv
    env = GatheringEnv(size=42,
                       num_players=2,
                       max_steps=STEPS_PER_EPISODE,
                       player_move_cost=1,
                       player_respawn_time=5,
                       player_murder_mode=MURDER_MODE,
                       food_reward=25,
                       food_respawn_time=40,
                       food_level=FOOD_LITTLE)
    return env


if __name__ == "__main__":
    ray.init()
        # redis_max_memory=5_000_000_000,
        # memory=5_000_000_000,
        # object_store_memory=5_000_000_000,
    register_env("gathering", env_creator)
    trainer0, trainer1 = get_player_trainers(evaluate=False, murder_mode=MURDER_MODE)
    for i in tqdm(range(NUM_EPISODES)):
        # improve first policy
        trainer0.train()
        # improve second policy
        trainer1.train()
        if i % CHECKPOIINT_INTERVAL == 0:
            checkpoint = trainer0.save()
            print("trainer 0  checkpoint saved at", checkpoint)
            checkpoint = trainer1.save()
            print("trainer 1 checkpoint saved at", checkpoint)

# swap weights to synchronize
# dqn_trainer.set_weights(ppo_trainer.get_weights(["ppo_policy"]))
# ppo_trainer.set_weights(dqn_trainer.get_weights(["dqn_policy"]))
# dqn_trainer1.set_weights(dqn_trainer2.get_weights(["2"]))
# dqn_trainer2.set_weights(dqn_trainer1.get_weights(["1"]))