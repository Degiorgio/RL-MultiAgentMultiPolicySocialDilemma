import os
import numpy as np
import random
import tensorflow as tf

import util
from conv_model import create_model
from tqdm import tqdm
from DQNAgent import DQNAgent
from agent_wrapper import agent_wrapper
from env.gridworld import BlobEnv
from env.gridworld import FOOD_TINY, FOOD_LITTLE, FOOD_ALOT, FOOD_NORMAL

import hyperparameters
from concurrent.futures import ThreadPoolExecutor

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# setup logging
executor_1 = ThreadPoolExecutor(max_workers=1)
executor_2 = ThreadPoolExecutor(max_workers=1)
output_path = os.path.join("out", util.get_run_id())
util.create_dir(output_path)
__logger = util.setup_log(os.path.join(output_path, "env.txt"))


def report(handle, episode, avg_reward, min_reward, max_reward, epsilon):
    executor_1.submit(
        util.__thread_report_log,
        (handle, episode, avg_reward, min_reward, max_reward, epsilon)
    )


def log(string):
    executor_2.submit(
        util.__thread_log,
        (__logger, string)
    )


def unwind(handles):
    executor_1.shutdown(shutdown=True)
    executor_2.shutdown(shutdown=True)
    for x in handles:
        handles.close()


def rl_loop(env, agent, NUM_EPISODES, STATS_EVERY, SHOW_PREVIEW):
    for episode in tqdm(range(1, NUM_EPISODES + 1)):
        step = 1
        done = False
        current_state = env.reset(num_players=1)
        while not done:
            action = agent.get_action(current_state)
            new_state, rewards, done = env.step([action])

            if SHOW_PREVIEW and not episode % STATS_EVERY:
                env.render()

            agent.do_step(
                step, rewards[0], current_state, action, new_state, done
            )

            current_state = new_state
            step += 1

        agent.episode_done(episode)

        if not episode % STATS_EVERY or episode == 1:
            agent.write_stats(episode, STATS_EVERY, report)


# SETUP ENVIRONMENT

env = BlobEnv(
    size=20,
    return_images=True,
    max_steps=300,
    player_respawn_time=20,
    player_murder_mode=False,
    food_respawn_time=30,
    food_level=FOOD_LITTLE,
    draw_shooting_direction=True,
    logger_callback=log)

observation_space = env.get_observation_space()
n_actions = env.get_n_actions()

print(f"observation space: {observation_space}")
print(f"number of actions: {n_actions}")

# SETUP DQN MODEL

main_model = create_model(observation_space, n_actions,
                          learning_rate=0.001, dropout=0.2)
target_model = create_model(observation_space, n_actions,
                            learning_rate=0.001, dropout=0.2)
RL_algor = DQNAgent(main_model, target_model,
                    REPLAY_MEMORY_SIZE=50_000,
                    MIN_REPLAY_MEMORY_SIZE=1_000,
                    MINIBATCH_SIZE=64,
                    DISCOUNT=0.99,
                    UPDATE_TARGET_EVERY=5)

# SETUP AGENTS

sf = util.get_stats_file(os.path.join(output_path, "player_1_stats.txt"))
agent = agent_wrapper(sf,
                      log, 1,
                      RL_algor, n_actions,
                      hyperparameters.p1_epsilon,
                      hyperparameters.p1_MIN_EPSILON,
                      hyperparameters.p1_EPSILON_DECAY)

# GO

rl_loop(env, agent,
        hyperparameters.EPISODES,
        hyperparameters.STATS_EVERY,
        hyperparameters.SHOW_PREVIEW)

# UNWIND

unwind([sf])
