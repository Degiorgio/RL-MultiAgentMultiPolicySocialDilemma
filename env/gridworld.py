import numpy as np
import cv2
from PIL import Image
from blob import Blob
import collections

Point = collections.namedtuple('Point', 'x y')

FOOD_TINY = 1
FOOD_LITTLE = 2
FOOD_NORMAL = 4
FOOD_ALOT = 7


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
    def __init__(self, pid, size, color, x=None, y=None, facing_dir=None):
        super(Player, self).__init__(size, x, y, facing_dir)
        self.pid = pid
        self.dead = False
        self.time_of_death = None
        self.color = color

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


class BlobEnv:
    def __init__(self,
                 size=20,
                 return_images=True,
                 max_steps=1000,
                 player_respawn_time=5,
                 player_murder_mode=True,
                 food_respawn_time=20,
                 food_level=FOOD_NORMAL,
                 draw_shooting_direction=True,
                 logger=None):
        self.size = size
        self.return_images = return_images
        self.max_steps = max_steps

        self.players = None
        self.player_move_cost = 1
        self.murder_mode = player_murder_mode
        self.player_respawn_time = player_respawn_time
        self.draw_shooting_direction = draw_shooting_direction

        self.food = None
        self.food_level = food_level
        self.food_respawn_time = food_respawn_time
        self.food_reward = 25
        self.logger = None

    def get_n_actions(self):
        if self.murder_mode:
            return 4
        else:
            return 5

    def generate_food(self, start=np.array([0, 0])):
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

    def _add_player(self, color):
        player = Player(len(self.players), self.size, color=color, facing_dir=0)
        while player in self.food or player in self.players:
            player = Player(len(self.players), self.size, color=color, facing_dir=0)
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

            if self.murder_mode and player.beam:
                if self.logger is not None:
                    self.logger.info(f"player: {player} shooting")
                player.beam = False
                for player2 in self.players:
                    if player.pid == player2.pid:
                        continue
                    if player2.hit_by(player):
                        if self.logger is not None:
                            self.logger.info(
                                f"[{player2}] was hit by [{player}]")
                        player2.died(self.episode_step)

        if self.murder_mode:
            # respawn players if appropriate
            for player in self.players:
                if player.dead:
                    elapsed_time = self.episode_step - player.time_of_death
                    if elapsed_time >= self.player_respawn_time:
                        player.respawn()
                        while player in self.food:
                            player.respawn()

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

    def contains_player(self, x, y, pid):
        point = Point(x=x, y=y)
        if point in self.players:
            pindex = self.players.index(point)
            if pindex == (pid-1) and not self.players[pindex].dead:
                return True
        else:
            return False

    def contains_player_direction(self, x, y):
        for player in self.players:
            if not player.dead:
                direction = player.get_facing()
                if x == direction[0] and direction[1] == y:
                    return True
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
                if self.murder_mode and self.draw_shooting_direction:
                    facing = player.get_facing()
                    env[facing[0]][facing[1]] = (128, 128, 128)

        # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        img = Image.fromarray(env, 'RGB')
        return img
