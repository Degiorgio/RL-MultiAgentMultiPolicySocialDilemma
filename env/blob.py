import numpy as np
import collections

Point = collections.namedtuple('Point', 'x y')

# actions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
NOTHING = 4

# this action is only valid with murder mode enabled
SHOOT = 5

# the followings actions are only valid with murder mode enabled and
# SHOOT_IN_ALL_DIRECTIONS set to false
ROTATE_LEFT = 6
ROTATE_RIGHT = 7

# facing directions
NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

action_string_map = {
    0: "move_up",
    1: "move_down",
    2: "move_left",
    3: "move_right",
    4: "nothing",
    5: "shoot",
    6: "rotate_left",
    7: "rotate_right"
}


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

    # COORDINATES ARE IMAGE BASED. I.E:
    # (x=SIZE,y=SIZE) = bottom right
    # and
    # (x=0, y=0) = top left
    def action(self, choice, murder_mode, obstructions, shoot_in_all_directions):
        if not murder_mode:
            if choice == UP:
                # move up
                return self.move(x=1, y=0, obstructions=obstructions)
                # self.move(x=0, y=-1)
            elif choice == DOWN:
                #  move down
                return self.move(x=-1, y=0, obstructions=obstructions)
                # self.move(x=0, y=1)
            elif choice == LEFT:
                # move left
                # self.move(x=-1, y=0)
                return self.move(x=0, y=-1, obstructions=obstructions)
            elif choice == RIGHT:
                # move right
                # self.move(x=1, y=0)
                return self.move(x=0, y=1, obstructions=obstructions)
            elif choice == NOTHING:
                return False
            else:
                raise NotImplementedError(f"action {choice} not supported")
        elif shoot_in_all_directions:
            if choice == UP:
                return self.move(x=1, y=0, obstructions=obstructions)
            elif choice == DOWN:
                return self.move(x=-1, y=0, obstructions=obstructions)
            elif choice == LEFT:
                return self.move(x=0, y=-1, obstructions=obstructions)
            elif choice == RIGHT:
                return self.move(x=0, y=1, obstructions=obstructions)
            elif choice == SHOOT:
                self.beam = True
                return False
            elif choice == NOTHING:
                return False
            else:
                raise NotImplementedError(f"action {choice} not supported")
        else:
            if choice == UP:
                # move up
                return self.move(x=1, y=0, obstructions=obstructions)
                # self.move(x=0, y=-1)
            elif choice == DOWN:
                #  move down
                return self.move(x=-1, y=0, obstructions=obstructions)
                # self.move(x=0, y=1)
            elif choice == LEFT:
                # move left
                # self.move(x=-1, y=0)
                return self.move(x=0, y=-1, obstructions=obstructions)
            elif choice == RIGHT:
                # move right
                # self.move(x=1, y=0)
                return self.move(x=0, y=1, obstructions=obstructions)
            elif choice == ROTATE_LEFT:
                self.facing_dir += 1
            elif choice == ROTATE_RIGHT:
                self.facing_dir -= 1
            elif choice == SHOOT:
                self.beam = True
            elif choice == NOTHING:
                return False
            else:
                raise NotImplementedError(f"action {choice} not supported")
        return False

    def move(self, x=None, y=None, obstructions=None):
        # if no value for x, move randomly
        p_x = self.x
        p_y = self.y
        if x is None:
            p_x += np.random.randint(-1, 2)
        else:
            p_x += x

        # if no value for y, move randomly
        if y is None:
            p_y += np.random.randint(-1, 2)
        else:
            p_y += y

        # make sure we are not obstructed
        point = Point(x=p_x, y=p_y)
        if point in obstructions:
            return True

        self.x = p_x
        self.y = p_y

        # if we are out of bounds, fix!
        if self.x < 1:
            self.x = 1
        elif self.x > self.size-2:
            self.x = self.size-2
        if self.y < 1:
            self.y = 1
        elif self.y > self.size-2:
            self.y = self.size-2

        return False

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

    def get_hit_intervals(self, shoot_in_all_directions):
        x_points = []
        y_points = []

        if not shoot_in_all_directions:
            direction = (self.facing_dir) % 4
        else:
            direction = None

        if direction == NORTH or direction is None:
            x_points += list(range(1, self.x))
            y_points += [self.y]*len(x_points)
        if direction == SOUTH or direction is None:
            x_points += list(range(self.x+1, self.size-1))
            y_points += [self.y]*len(x_points)
        if direction == WEST or direction is None:
            y_points += list(range(1, self.y))
            x_points += [self.x]*len(y_points)
        if direction == EAST or direction is None:
            y_points += list(range(self.y+1, self.size-1))
            x_points += [self.x]*len(y_points)
        points = list(zip(x_points, y_points))
        return points

    def set_hit(self, array, value, shoot_in_all_directions):
        if not shoot_in_all_directions:
            direction = (self.facing_dir) % 4
            if direction == NORTH:
                array[1:self.x, self.y] = value
                return array
            if direction == SOUTH:
                array[self.x+1:self.size-1, self.y] = value
                return array
            elif direction == WEST:
                array[self.x, 1:self.y] = value
                return array
            elif direction == EAST:
                array[self.x, self.y+1:self.size-1] = value
                return array
        else:
            direction = None
            array[1:self.x, self.y] = value
            array[self.x+1:self.size-1, self.y] = value
            array[self.x, 1:self.y] = value
            array[self.x, self.y+1:self.size-1] = value
            return array

    def hit_by(self, other, shoot_in_all_directions):
        hit = False
        if not shoot_in_all_directions:
            direction = (other.facing_dir) % 4
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
        else:
            if other.y == self.y and self.x <= other.x:
                hit = True
            elif other.y == self.y and self.x >= other.x:
                hit = True
            elif other.y >= self.y and self.x == other.x:
                hit = True
            elif other.y <= self.y and self.x == other.x:
                hit = True
        return hit
