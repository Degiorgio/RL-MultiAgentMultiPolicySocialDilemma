from env.gridworld import FOOD_TINY, FOOD_LITTLE, FOOD_ALOT, FOOD_NORMAL

# rl loop settings

EPISODES = 20_000
MIN_REWARD = -200

# stat reporting SETTINGS

STATS_EVERY = 50
SHOW_PREVIEW = False

# exploration settings - player 1

p1_epsilon = 1                      # not a constant, going to be decayed
p1_EPSILON_DECAY = 0.99975
p1_MIN_EPSILON = 0.001

# enviroment settings

SIZE = 20
MAX_STEPS_PER_EPISODE = 300
PLAYER_RESPAWN_TIME = 20
PLAYER_MURDER_MODE = True
FOOD_RESPAWN_TIME = 30
FOOD_LEVEL = FOOD_LITTLE
DRAW_SHOOTING_DIRECTION = True


def get_DQN_Algo(observation_space, n_actions):
    from conv_model import create_model
    from DQNAgent import DQNAgent
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
    return RL_algor
