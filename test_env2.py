from env.gridworld import GatheringEnv

num_players=2

if __name__ == "__main__":
    env = GatheringEnv(num_players=num_players)
    initial_state = env.reset()
    print(initial_state.keys())
    for i_episode in range(2):
        for t in range(10):
            actions = {}
            for x in  range(num_players):
                action = env.action_space.sample()
                actions[f"player{x}"] = action
            print(actions)
            observation, reward, done, info = env.step(actions)
            # print(observation.keys())
            # print(reward)
