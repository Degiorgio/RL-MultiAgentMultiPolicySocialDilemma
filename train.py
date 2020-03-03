from tqdm import tqdm
from configs import get_player_trainers
from experiments_configs import experiment_DQN_DQN, experiment_scarse, experiment_very_scarse

NUM_EPISODES = 2_500
CHECKPOIINT_INTERVAL = 50

# EXPERIMENT_CONFIGS = experiment_DQN_DQN(experiment_very_scarse, "TestForShooting_very_scare")
EXPERIMENT_CONFIGS = experiment_DQN_DQN(experiment_very_scarse(), "TestForShooting_scare")


if __name__ == "__main__":
    print("-----------------------------------------")
    print("configs")
    import json
    print(json.dumps(EXPERIMENT_CONFIGS, indent=4, sort_keys=True))
    print("-----------------------------------------")
    trainer0, trainer1 = get_player_trainers(
        False,
        EXPERIMENT_CONFIGS
    )
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
# redis_max_memory=5_000_000_000,
# memory=5_000_000_000,
# object_store_memory=5_000_000_000,
