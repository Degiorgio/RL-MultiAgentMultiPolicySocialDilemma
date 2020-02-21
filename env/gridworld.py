import numpy as np
import cv2
from PIL import Image
from blob import Blob
import collections

Point = collections.namedtuple('Point', 'x y')


class Apple(Blob):
    def __init__(self, size, x=None, y=None):
        self.collected = False
        self.collected_time = None
        self.color = (0, 255, 0)  # green
        super(Apple, self).__init__(size, x, y)

    def collect(self, step):
        self.collected = True
        self.collected_time = step

    def respawn(self):
        self.collected = False
        self.collected_time = None


class Player(Blob):
    def __init__(self, size, color, x=None, y=None):
        self.dead = False
        self.time_of_death = None
        self.color = color
        super(Player, self).__init__(size, x, y)

    def died(self, step):
        self.dead = True
        self.time_of_death = step


class BlobEnv:
    def __init__(self, size=20, return_images=True, max_steps=200):
        self.size = size
        self.return_images = return_images
        self.max_steps = max_steps

        self.player_move_cost = 1

        self.food_respawn_time = 3
        self.food_reward = 25

        self.players = None
        self.food = None

    def generate_food(self, size=3, start=np.array([0, 0])):
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

    def _add_player(self, color):
        player = Player(self.size, color=color)
        while player in self.food or player in self.players:
            player = Player(self.size, color=color)
        self.players.append(player)

    def reset(self, num_players=2):
        self.players = []
        self.food = []
        self.episode_step = 0

        # generate food
        self.generate_food(
            start=np.array([(self.size//2)-5, (self.size//2)-5]))

        # generate players
        self._add_player(color=(255, 0, 0))
        if num_players == 2:
            self._add_player(color=(0, 0, 255))
        elif num_players != 1:
            raise NotImplementedError("only 1-2 players supported")

        if self.return_images:
            observation = np.array(self.get_image())

        return observation

    def step(self, actions):
        assert(len(actions) == len(self.players))
        self.episode_step += 1

        rewards = [0 for x in range(len(self.players))]
        for player_index, player in enumerate(self.players):
            player.action(actions[player_index])
            if player in self.food:
                food_index = self.food.index(player)
                self.food[food_index].collect(self.episode_step)
                rewards[player_index] += self.food_reward
            else:
                rewards[player_index] += -self.player_move_cost

        # re-spawn food if appropriate
        for apple in self.food:
            if apple.collected:
                elapsed_time = self.episode_step - apple.collected_time
                if elapsed_time >= self.food_respawn_time:
                    apple.respawn()

        # compute new_observation
        if self.return_images:
            new_observation = np.array(self.get_image())

        # check if game is finished
        done = False
        if self.episode_step >= self.max_steps:
            done = True

        return new_observation, rewards, done

    def render(self, wait=True):
        img = self.get_image()
        # resizing so we can see
        img = img.resize((300, 300))
        cv2.imshow("image", np.array(img))
        if wait:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def contains_food(self, x, y):
        point = Point(x=x, y=y)
        if point in self.food:
            food_index = self.food.index(point)
            if not self.food[food_index].collected:
                return True
        return False

    def contains_player(self, x, y, player):
        point = Point(x=x, y=y)
        if point in self.players:
            pindex = self.players.index(point)
            if pindex == player-1 and not self.players[pindex].dead:
                return True
        else:
            return False

    def get_image(self):
        # starts an rbg of our size
        env = np.zeros((self.size, self.size, 3), dtype=np.uint8)

        # draw food
        for apple in self.food:
            if not apple.collected:
                env[apple.x][apple.y] = apple.color

        # draw players
        for player in self.players:
            if not player.dead:
                env[player.x][player.y] = player.color

        # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        img = Image.fromarray(env, 'RGB')
        return img
