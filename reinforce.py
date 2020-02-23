import os
import random

import numpy as np
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import util
from agent_wrapper import agent_wrapper
from env.gridworld import BlobEnv


import hyperparameters

run_id = util.get_run_id()

print(run_id)

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# setup logging
executor_1 = ThreadPoolExecutor(max_workers=1)
executor_2 = ThreadPoolExecutor(max_workers=1)
output_path = os.path.join("out", "test")
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


def rl_loop(env, agents, NUM_EPISODES, STATS_EVERY, SHOW_PREVIEW):
    for episode in tqdm(range(1, NUM_EPISODES + 1)):
        step = 1
        done = False
        current_state = env.reset(num_players=len(agents))
        while not done:
            actions = [agent.get_action(current_state) for agent in agents]
            new_state, rewards, done = env.step(actions)

            if SHOW_PREVIEW and not episode % STATS_EVERY:
                env.render()

            for i, agent in enumerate(agents):
                agent.do_step(
                    rewards[i], current_state, actions[i], new_state, done
                )
            current_state = new_state
            step += 1

        for agent in agents:
            agent.episode_done(episode)
            if not episode % STATS_EVERY or episode == 1:
                agent.write_stats(episode, STATS_EVERY, report)


def rl_loop_multiple(envs, agents, NUM_EPISODES, STATS_EVERY, SHOW_PREVIEW):
    executors = ThreadPoolExecutor(max_workers=40)

# SETUP ENVIRONMENT

env = BlobEnv(
    size=hyperparameters.SIZE,
    return_images=True,
    max_steps=hyperparameters.MAX_STEPS_PER_EPISODE,
    player_respawn_time=hyperparameters.PLAYER_RESPAWN_TIME,
    player_murder_mode=hyperparameters.PLAYER_MURDER_MODE,
    food_respawn_time=hyperparameters.FOOD_RESPAWN_TIME,
    food_level=hyperparameters.FOOD_LEVEL,
    draw_shooting_direction=hyperparameters.DRAW_SHOOTING_DIRECTION,
    logger_callback=log)

observation_space = env.get_observation_space()
n_actions = env.get_n_actions()

print(f"observation space: {observation_space}")
print(f"number of actions: {n_actions}")

# SETUP AGENTS

sf1 = util.get_stats_file(os.path.join(output_path, "player_1_stats.txt"))
sf2 = util.get_stats_file(os.path.join(output_path, "player_2_stats.txt"))
agent1 = agent_wrapper(sf1, log, 1,
                       hyperparameters.get_DQN_Algo(observation_space, n_actions),
                       n_actions,
                       hyperparameters.p1_epsilon,
                       hyperparameters.p1_MIN_EPSILON,
                       hyperparameters.p1_EPSILON_DECAY)
agent2 = agent_wrapper(sf2, log, 2,
                       hyperparameters.get_DQN_Algo(observation_space, n_actions),
                       n_actions,
                       hyperparameters.p1_epsilon,
                       hyperparameters.p1_MIN_EPSILON,
                       hyperparameters.p1_EPSILON_DECAY)

# GO

rl_loop(env, [agent1, agent2],
        hyperparameters.EPISODES,
        hyperparameters.STATS_EVERY,
        hyperparameters.SHOW_PREVIEW)

# UNWIND

unwind([sf1, sf2])
