from multiprocessing import Process
import multiprocessing as mp
import sys
import os
import json
import glob
from tqdm import tqdm
import numpy as np
from collections import Counter
from configs import get_player_trainers, env_creator
from util import create_dir
import statistics

def _render_video(image_path):
    # TODO
    return

def _save_image(step, img, experiment_path):
    img_path = os.path.join(experiment_path, "render")
    create_dir(img_path, clean=False)
    if step == 0:
        files = glob.glob(f'{img_path}/*')
        for f in files:
            os.remove(f)
    img_path = os.path.join(img_path, f"{str(step).zfill(5)}.png")
    img = img.resize((1000, 1000))
    img.save(img_path)


def _play_game(trainerplayer0,
               trainerplayer1,
               env,
               steps_per_episode,
               episodes,
               save_images,
               experiment_path,
               render_video):
    from env.blob import action_string_map
    from env.gridworld import foot_level_string_map
    print("-----------------------------------------")

    player0_action_distribution = \
        dict.fromkeys(action_string_map.keys(), 0)
    player1_action_distribution = \
        dict.fromkeys(action_string_map.keys(), 0)

    average_reward = Counter({"player0": 0, "player1": 0})
    player0_rewards = []
    player1_rewards = []
    for episode in tqdm(range(episodes)):
        current_state = env.reset()
        step = 0
        cum_rewards = Counter({"player0": 0, "player1": 0})
        if episode == 0 and save_images:
            _save_image(step, env._get_image(), experiment_path)
        while True:
            step += 1
            if trainerplayer1 is None:
                # Use Random Policy
                actions = {
                    "player0": trainerplayer0.compute_action(
                        current_state["player0"],
                        policy_id="0",
                        explore=False),
                    "player1": np.random.randint(0, env._get_action_space())
                }
            else:
                actions = {
                    "player0": trainerplayer0.compute_action(
                        current_state["player0"],
                        policy_id="0",
                        explore=False),
                    "player1": trainerplayer1.compute_action(
                        current_state["player1"],
                        policy_id="1",
                        explore=False)
                }
            player0_action_distribution[actions["player0"]] += 1
            player1_action_distribution[actions["player1"]] += 1
            new_observaton, rewards, done, info = env.step(actions)
            if episode == 0 and save_images:
                _save_image(step, env._get_image(), experiment_path)
            cum_rewards.update(rewards)
            current_state = new_observaton
            if done['__all__']:
                break
        # print("rewards",  cum_rewards)
        average_reward.update(cum_rewards)
        player0_rewards.append(cum_rewards['player0'])
        player1_rewards.append(cum_rewards['player1'])
        if render_video and episode == 0:
            _render_video(os.path.join(experiment_path, "render"))
    print("-----------------------------------------")
    player0_action_distribution = \
        {action_string_map[k]: ((v/episodes)/steps_per_episode) for k, v in player0_action_distribution.items()}
    player1_action_distribution = \
        {action_string_map[k]: ((v/episodes)/steps_per_episode) for k, v in player1_action_distribution.items()}

    print("action distribution player 0", player0_action_distribution)
    print("action distribution player 1", player1_action_distribution)

    median = {"player0": 0, "player1": 0}
    median['player0'] = statistics.median(player0_rewards)
    median['player1'] = statistics.median(player1_rewards)
    average_reward['player0'] = average_reward['player0']/episodes
    average_reward['player1'] = average_reward['player1']/episodes
    print("average reward:", average_reward)
    print("median reward:", median)

    with open(os.path.join(experiment_path, "results_eval.json"), "w") as f:
        results = {
            "player0_action_distribution": player0_action_distribution,
            "player1_action_distribution": player1_action_distribution,
            "player0_reward": average_reward['player0'],
            "player1_reward": average_reward['player1'],
            "player0_median": median['player0'],
            "player1_median": median['player1'],
            "number_of_episodes": episodes
        }
        json.dump(results, f, indent=4)


def evaluate(experiment_path,
             NUM_EPISODES,
             player_0_checkpoint_index,
             player_1_checkpoint_index,
             USE_RANDOM_POLICY_FOR_PLAYER_1,
             save_images,
             render_video):
    # setup paths
    experiment_params_path = os.path.join(experiment_path, "experiment.json")
    player1_path = os.path.join(experiment_path, "player1")
    player0_path = os.path.join(experiment_path, "player0")

    parameters_player0_path = os.path.join(player0_path, "params.json")
    parameters_player1_path = os.path.join(player1_path, "params.json")

    assert(os.path.exists(experiment_params_path))
    assert(os.path.exists(parameters_player0_path))
    assert(os.path.exists(parameters_player1_path))

    checkpoints_player0 = glob.glob(
        os.path.join(player0_path, "checkpoint_*"),
        recursive=False)
    checkpoints_player1 = glob.glob(
        os.path.join(player1_path, "checkpoint_*"),
        recursive=False)

    checkpoints_player0.sort(
        key=lambda x: int(os.path.basename(x).split('.')[0][11:]))
    checkpoints_player1.sort(
        key=lambda x: int(os.path.basename(x).split('.')[0][11:]))

    model_path_player0 = os.path.join(
        checkpoints_player0[player_0_checkpoint_index],
        "checkpoint-"+os.path.basename(checkpoints_player0[player_0_checkpoint_index])[11:])

    model_path_player1 = os.path.join(
        checkpoints_player1[player_1_checkpoint_index],
        "checkpoint-"+os.path.basename(checkpoints_player1[player_1_checkpoint_index])[11:])

    assert(os.path.exists(model_path_player0))
    assert(os.path.exists(model_path_player1))

    print("-----------------------------------------")
    print(f"using checkpoint {model_path_player0} for player 0")
    print(f"using checkpoint {model_path_player1} for player 1")

    # load params used to train
    with open(experiment_params_path, 'r') as f:
        experiment_params = json.load(f)

    env_config = experiment_params['env_configs']
    print("-----------------------------------------")
    print("configs")
    print(json.dumps(experiment_params, indent=4, sort_keys=True))
    print("-----------------------------------------")

    trainer0, trainer1 = get_player_trainers(True, experiment_params)
    trainer0.restore(model_path_player0)
    trainer1.restore(model_path_player1)

    steps_per_episode = env_config["steps_per_episode"]
    env = env_creator(env_config=env_config)

    if USE_RANDOM_POLICY_FOR_PLAYER_1:
        print("using random policy for player 1")
        _play_game(trainer0,
                   None,
                   env=env, episodes=NUM_EPISODES,
                   steps_per_episode=steps_per_episode,
                   save_images=save_images,
                   experiment_path=experiment_path,
                   render_video=render_video)
    else:
        _play_game(trainer0, trainer1,
                   env=env,
                   episodes=NUM_EPISODES,
                   steps_per_episode=steps_per_episode,
                   save_images=save_images,
                   experiment_path=experiment_path,
                   render_video=render_video)


# experiment_path = sys.argv[1]
# NUM_EPISODES = int(sys.argv[2])
def main(experiment_path, NUM_EPISODES):
    print(f"evaluating {experiment_path}")
    # optional configs
    player_0_checkpoint_index = 4
    player_1_checkpoint_index = 4
    USE_RANDOM_POLICY_FOR_PLAYER_1 = False
    save_images = True
    render_video = True
    evaluate(experiment_path,
             NUM_EPISODES,
             player_0_checkpoint_index,
             player_1_checkpoint_index,
             USE_RANDOM_POLICY_FOR_PLAYER_1,
             save_images,
             render_video)



folders = glob.glob("en3/*/")
processes = []
for folder in folders:
    # if not os.path.exists(os.path.join(folder, "results_eval.json")):
        p = Process(target=main, args=(folder, 100))
        p.start()
        processes.append(p)

mp.set_start_method('spawn')
for x in processes:
    x.join()

