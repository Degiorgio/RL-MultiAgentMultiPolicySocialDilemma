import json
import multiprocessing as mp

from agents import get_player_trainers


def work(x, EXPERIMENT_CONFIGS):
    print(x)
    expr = EXPERIMENT_CONFIGS[x]
    NUM_EPISODES = 2000
    CHECKPOIINT_INTERVAL = 50

    print("-----------------------------------------")
    print("configs")
    print(json.dumps(expr, indent=4, sort_keys=True))
    print("-----------------------------------------")

    trainer0, trainer1 = get_player_trainers(
        False,
        expr,
    )

    for i in range(NUM_EPISODES):
        # improve first policy
        trainer0.train()
        # improve second policy
        trainer1.train()
        if i % CHECKPOIINT_INTERVAL == 0:
            checkpoint = trainer0.save()
            print("trainer 0  checkpoint saved at", checkpoint)
            checkpoint = trainer1.save()
            print("trainer 1 checkpoint saved at", checkpoint)


def main():
    from experiments_configs import expr_environment
    from experiments_configs import expr_exploration_vs_exploitation
    from experiments_configs import expr_parameters
    processes = []
    EXPERIMENT_CONFIGS = expr_environment()
    for i, x in enumerate(EXPERIMENT_CONFIGS):
        p = mp.Process(target=work, args=(i,EXPERIMENT_CONFIGS))
        p.start()
        processes.append(p)
    for x in processes:
        x.join()


if __name__ == "__main__":
    main()

# If we want to synchronize the wieghts (which we dont here) we would need
# to swap the weights as follows:
#
# dqn_trainer.set_weights(ppo_trainer.get_weights(["ppo_policy"]))
# ppo_trainer.set_weights(dqn_trainer.get_weights(["dqn_policy"]))
# dqn_trainer1.set_weights(dqn_trainer2.get_weights(["2"]))
# dqn_trainer2.set_weights(dqn_trainer1.get_weights(["1"]))
