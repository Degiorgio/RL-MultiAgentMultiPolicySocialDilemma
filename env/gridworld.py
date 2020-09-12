import numpy as np

import cv2
from PIL import Image
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Discrete, Box

from env.blob import Blob, Point

# AMOUNT of food on the grid word

FOOD_TINY = 1
FOOD_LITTLE = 2
FOOD_NORMAL = 4
FOOD_ALOT = 7

foot_level_string_map = {
    1: "tiny",
    2: "little",
    4: "normal",
    7: "alot"
}


class Apple(Blob):
    def __init__(self, size, x=None, y=None):
        super(Apple, self).__init__(size, x, y)

        self.collected = False
        self.collected_time = None
        self.color = (0, 255, 0)

    def collect(self, step):
        self.collected = True
        self.collected_time = step

    def respawn(self):
        self.collected = False
        self.collected_time = None


class Player(Blob):
    def __init__(self, pid, size, beam_color, color, x=None, y=None, facing_dir=None):
        super(Player, self).__init__(size, x, y, facing_dir)
        self.pid = pid
        self.dead = False
        self.time_of_death = None
        self.color = color
        self.draw_beam = False
        self.beam_color = beam_color

    def died(self, step):
        self.dead = True
        self.time_of_death = step

    def respawn(self):
        self.dead = False
        self.time_of_death = None
        self.relocate()

    def __str__(self):
        return \
            f"Player {self.pid}: loc: ({self.x}, {self.y})"\
            f" dead: {self.dead}, facing_dir: {self.facing_dir % 4}"

    def __repr__(self):
        return \
            f"Player {self.pid}: loc: ({self.x}, {self.y}), dead:"\
            f" {self.dead}, facing_dir: {self.facing_dir % 4}"


class GatheringEnv(MultiAgentEnv):
    metadata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self,
                 size,
                 num_players,
                 steps_per_episode,
                 player_move_cost,
                 player_respawn_time,
                 murder_mode,
                 food_reward,
                 food_respawn_time,
                 food_level,
                 beam_color_diff,
                 draw_beam,
                 randomized_food,
                 small_world,
                 shoot_in_all_directions,
                 draw_shooting_direction):
        if small_world:
            self.size = 20
        else:
            self.size = size

        self.max_steps = steps_per_episode
        self.num_players = num_players

        self.players = None
        self.player_move_cost = player_move_cost
        self.murder_mode = murder_mode
        self.player_respawn_time = player_respawn_time
        self.draw_shooting_direction = draw_shooting_direction
        self.draw_beam = draw_beam
        self.beam_color_diff = beam_color_diff
        self.randomized_food = randomized_food

        self.shoot_in_all_directions = shoot_in_all_directions

        if self.shoot_in_all_directions:
            self.draw_shooting_direction = False

        self.small_world = small_world

        self.food = None
        self.food_level = food_level
        self.food_respawn_time = food_respawn_time
        self.food_reward = food_reward

        # GYM stuff
        self.action_space = Discrete(self._get_action_space())
        self.observation_space = Box(0, 255, [size, size, 3])

    def reset(self):
        self.players = []
        self.food = []
        self.episode_step = 0

        # generate food
        self._generate_food(
            start=np.array([(self.size//2)-5, (self.size//2)-5]))

        # generate players
        self._add_player(beam_c=(255, 255, 255), color=(255, 0, 0))
        if self.num_players == 2:
            if self.beam_color_diff:
                self._add_player(beam_c=(255, 192, 203), color=(0, 0, 255))
            else:
                self._add_player(beam_c=(255, 255, 255), color=(0, 0, 255))
        elif self.num_players != 1:
            raise NotImplementedError("only 1-2 players supported")

        observations = {}
        observation = np.array(self._get_image())
        for i, player in enumerate(self.players):
            observations["player"+str(i)] = observation.copy()
        return observations

    def step(self, actions):
        assert(len(actions) == len(self.players))
        self.episode_step += 1
        rewards = [0 for x in range(len(self.players))]

        # collect foood
        for player_index, player in enumerate(self.players):
            player.draw_beam = False
            if player.dead:
                rewards[player_index] += -self.player_move_cost
                continue

            obstructions = []
            for x in self.players:
                if x.dead:
                    continue
                if player.pid == x.pid:
                    continue
                obstructions.append(x)

            obstructed = player.action(
                actions[f"player{player_index}"],
                self.murder_mode,
                obstructions,
                self.shoot_in_all_directions)

            if obstructed:
                rewards[player_index] += -(self.player_move_cost)
                continue

            if self.murder_mode and player.beam:
                player.beam = False
                player.draw_beam = True
                for player2 in self.players:
                    if player.pid == player2.pid:
                        continue
                    if player2.hit_by(player,
                                      self.shoot_in_all_directions):
                        player2.died(self.episode_step)

            if player in self.food:
                food_index = self.food.index(player)
                if self.food[food_index].collected:
                    rewards[player_index] += -self.player_move_cost
                else:
                    self.food[food_index].collect(self.episode_step)
                    rewards[player_index] += self.food_reward
            else:
                rewards[player_index] += -self.player_move_cost

        rewards = {f"player{i}": reward for i, reward in enumerate(rewards)}
        alives = []

        if self.murder_mode:
            # respawn players if appropriate
            for player in self.players:
                if player.dead:
                    elapsed_time = self.episode_step - player.time_of_death
                    if elapsed_time >= self.player_respawn_time:
                        player.respawn()
                        while player in self.food:
                            player.respawn()
                        alives.append(player)
                else:
                    alives.append(player)


        # re-spawn food if appropriate
        for apple in self.food:
            if apple.collected:
                elapsed_time = self.episode_step - apple.collected_time
                if elapsed_time >= self.food_respawn_time:
                    if apple not in alives:
                        apple.respawn()

        # compute new_observation
        new_observations = {}
        new_observation = np.array(self._get_image())
        for i, player in enumerate(self.players):
            new_observations["player"+str(i)] = new_observation.copy()

        # check if game is finished
        done = {
            "__all__": self.episode_step >= self.max_steps
        }
        return new_observations, rewards, done, {}

    def render(self, mode="human", close=False):
        if mode == 'rgb_array':
            img = self._get_image()
            return img
        elif mode == "human":
            if close:
                cv2.destroyAllWindows()
            else:
                img = self._get_image()
                cv2.imshow("image", np.array(img))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            raise Exception("unsupported render mode: " + mode)

    def _get_action_space(self):
        if self.murder_mode:
            if not self.shoot_in_all_directions:
                return 8
            else:
                return 6
        else:
            return 5

    def _generate_food(self, start=np.array([0, 0])):
        size = self.food_level
        tl = size * 2 - 1
        top = start + tl - 1
        for idx in range(size - 1):
            for i in range(idx * 2 + 1):
                row = top[0] - idx
                col = start[1] + size - 1 - idx + i
                self.food.append(Apple(size=self.size, x=row, y=col))
        for idx in range(size - 1, -1, -1):
            for i in range(idx * 2 + 1):
                row = start[0] + idx
                col = start[1] + size - 1 - idx + i
                self.food.append(Apple(size=self.size, x=row, y=col))

    def _add_player(self, beam_c, color):
        player = Player(
            len(self.players), self.size, beam_color=beam_c, color=color, facing_dir=0
        )
        while player in self.food or player in self.players:
            player = Player(
                len(self.players), self.size, beam_color=beam_c, color=color, facing_dir=0
            )
        self.players.append(player)

    def _contains_food(self, x, y):
        point = Point(x=x, y=y)
        if point in self.food:
            food_index = self.food.index(point)
            if not self.food[food_index].collected:
                return True
        return False

    def _contains_player(self, x, y, pid):
        point = Point(x=x, y=y)
        if point in self.players:
            pindex = self.players.index(point)
            if pindex == (pid-1) and not self.players[pindex].dead:
                return True
        else:
            return False

    def _contains_player_direction(self, x, y):
        if self.murder_mode and self.draw_shooting_direction:
            for player in self.players:
                if not player.dead:
                    direction = player.get_facing()
                    if x == direction[0] and direction[1] == y:
                        return True
        return False

    def _contains_beam(self, x, y):
        # draw beam
        point = Point(x=x, y=y)
        for player in self.players:
            if player.draw_beam:
                points = player.get_hit_intervals(self.shoot_in_all_directions)
                if point in points:
                    return True
        return False

    def _get_image(self):
        # starts an rbg of our size
        env = np.zeros((self.size, self.size, 3), dtype=np.uint8)

        # draw beam
        if self.murder_mode and self.draw_beam:
            for player in self.players:
                if not player.dead:
                    if player.draw_beam:
                        env = player.set_hit(env, player.beam_color,
                                             self.shoot_in_all_directions)

        # shooting direction
        if self.murder_mode and self.draw_shooting_direction:
            for player in self.players:
                if not player.dead:
                    facing = player.get_facing()
                    env[facing[0]][facing[1]] = (128, 128, 128)
        # draw food
        for apple in self.food:
            if not apple.collected:
                env[apple.x][apple.y] = apple.color

        # draw players
        for player in self.players:
            if not player.dead:
                env[player.x][player.y] = player.color

        if self.small_world:
            env = np.pad(env, pad_width=((11, 11,), (11, 11,), (0, 0,)),
                         mode='constant', constant_values=0)

        img = Image.fromarray(env, 'RGB')
        return img
