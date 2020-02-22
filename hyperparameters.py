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
