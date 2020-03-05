import multiprocessing as mp

def work(x):
    print(x)
    from experiments_configs import expr_scrase
    from configs import get_player_trainers
    EXPERIMENT_CONFIGS = expr_scrase()
    expr = EXPERIMENT_CONFIGS[x]
    NUM_EPISODES = 500
    CHECKPOIINT_INTERVAL = 50
    print("-----------------------------------------")
    print("configs")
    import json
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
    return

def main():
    processes = []
    from experiments_configs import expr_scrase
    EXPERIMENT_CONFIGS = expr_scrase()
    for i, x in enumerate(EXPERIMENT_CONFIGS):
        p = mp.Process(target=work, args=(i,))
        p.start()
        processes.append(p)
    for x in processes:
        x.join()

main()


# swap weights to synchronize
# dqn_trainer.set_weights(ppo_trainer.get_weights(["ppo_policy"]))
# ppo_trainer.set_weights(dqn_trainer.get_weights(["dqn_policy"]))
# dqn_trainer1.set_weights(dqn_trainer2.get_weights(["2"]))
# dqn_trainer2.set_weights(dqn_trainer1.get_weights(["1"]))
# redis_max_memory=5_000_000_000,
# memory=5_000_000_000,
# object_store_memory=5_000_000_000,
