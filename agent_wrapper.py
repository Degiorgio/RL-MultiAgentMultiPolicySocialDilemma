import numpy as np


class agent_wrapper:
    def __init__(self,
                 stats_file,
                 logger_callback,  index,
                 agent, n_actions, epsilon, MIN_EPSILON, EPSILON_DECAY):
        self.episode_reward = 0
        self.episode_rewards = []

        self.index = index
        self.agent = agent
        self.n_actions = n_actions

        self.epsilon = epsilon
        self.MIN_EPSILON = MIN_EPSILON
        self.EPSILON_DECAY = EPSILON_DECAY

        self.logger_callback = logger_callback
        self.stats_file = stats_file

        self.logger_callback(
            f"Agent {self.index} HP:" +
            f" initial epsilon {self.epsilon}," +
            f" minimum epsilon {self.MIN_EPSILON},"
            f" epsilon decay {self.EPSILON_DECAY},"
        )

    def get_action(self, current_state):
        if np.random.random() > self.epsilon:
            # Get action from Q table
            action = np.argmax(self.agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, self.n_actions)
        return action

    def episode_done(self, episode):
        self.episode_rewards.append(self.episode_reward)
        # Decay epsilon
        if self.epsilon > self.MIN_EPSILON:
            self.epsilon *= self.EPSILON_DECAY
            self.epsilon = max(self.MIN_EPSILON, self.epsilon)

        self.logger_callback(
            f"episode: {episode} done, agent {self.index} reward: {self.episode_reward}"
        )

        self.episode_reward = 0

    def do_step(self, reward, current_state, action, new_state, done):
        self.episode_reward += reward
        self.agent.update_replay_memory(
            (current_state, action, reward, new_state, done)
        )
        self.agent.train(done)

    def write_stats(self, episode, EVERY, cb):
        span = self.episode_rewards[-EVERY:]
        average_reward = sum(span)/len(span)
        min_reward = min(span)
        max_reward = max(span)
        cb(self.stats_file, episode,
           average_reward, min_reward, max_reward,
           self.epsilon)
