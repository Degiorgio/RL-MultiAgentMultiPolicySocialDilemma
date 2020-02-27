import numpy as np

# actions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
NOTHING = 4

# murder mode only
ROTATE_LEFT = 5
ROTATE_RIGHT = 6
SHOOT = 7

# facing directions
NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3


class Blob:
    def __init__(self, size, x=None, y=None, facing_dir=None):
        self.size = size
        self.beam = False
        if facing_dir is None:
            self.facing_dir = np.random.randint(0, 4)
        else:
            self.facing_dir = facing_dir
        if x is None:
            self.x = np.random.randint(1, size-1)
        else:
            self.x = x
        if y is None:
            self.y = np.random.randint(1, size-1)
        else:
            self.y = y

    def __str__(self):
        return f"blob ({self.x}, {self.y})"

    def __repr__(self):
        return f"blob ({self.x}, {self.y})"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def relocate(self):
        self.x = np.random.randint(1, self.size-1)
        self.y = np.random.randint(1, self.size-1)
        self.facing_dir = np.random.randint(0, 4)

    # COORDINATES ARE IMAGE BASED. I.e:
    # (x=SIZE,y=SIZE)=bottom rigth
    # and
    # (x=0, y=0)=top left
    def action(self, choice, murder_mode):
        if not murder_mode:
            if choice == UP:
                # move up
                self.move(x=1, y=0)
                # self.move(x=0, y=-1)
            elif choice == DOWN:
                #  move down
                self.move(x=-1, y=0)
                # self.move(x=0, y=1)
            elif choice == LEFT:
                # move left
                # self.move(x=-1, y=0)
                self.move(x=0, y=-1)
            elif choice == RIGHT:
                # move right
                # self.move(x=1, y=0)
                self.move(x=0, y=1)
            elif choice == NOTHING:
                return
            else:
                raise NotImplementedError(f"action {choice} not supported")
        else:
            if choice == UP:
                # move up
                self.move(x=1, y=0)
                # self.move(x=0, y=-1)
            elif choice == DOWN:
                #  move down
                self.move(x=-1, y=0)
                # self.move(x=0, y=1)
            elif choice == LEFT:
                # move left
                # self.move(x=-1, y=0)
                self.move(x=0, y=-1)
            elif choice == RIGHT:
                # move right
                # self.move(x=1, y=0)
                self.move(x=0, y=1)
            elif choice == ROTATE_LEFT:
                self.facing_dir += 1
            elif choice == ROTATE_RIGHT:
                self.facing_dir -= 1
            elif choice == SHOOT:
                self.beam = True
            elif choice == NOTHING:
                return
            else:
                raise NotImplementedError(f"action {choice} not supported")

    def move(self, x=None, y=None):
        # if no value for x, move randomly
        if x is None:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # if no value for y, move randomly
        if y is None:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # if we are out of bounds, fix!
        if self.x < 1:
            self.x = 1
        elif self.x > self.size-2:
            self.x = self.size-2
        if self.y < 1:
            self.y = 1
        elif self.y > self.size-2:
            self.y = self.size-2

    def get_facing(self):
        direction = (self.facing_dir) % 4
        if direction == NORTH:
            return (self.x-1, self.y)
        elif direction == SOUTH:
            return (self.x+1, self.y)
        elif direction == WEST:
            return (self.x, self.y-1)
        elif direction == EAST:
            return (self.x, self.y+1)
        else:
            raise NotImplementedError(
                f"facing {self.facing_dir} not supported"
            )

    def hit_by(self, other):
        direction = (other.facing_dir) % 4
        hit = False
        if direction == NORTH:
            if other.y == self.y and self.x <= other.x:
                hit = True
        elif direction == SOUTH:
            if other.y == self.y and self.x >= other.x:
                hit = True
        elif direction == WEST:
            if other.y >= self.y and self.x == other.x:
                hit = True
        elif direction == EAST:
            if other.y <= self.y and self.x == other.x:
                hit = True
        return hit
