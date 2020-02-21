import numpy as np

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
NOTHING = 4


class Blob:
    def __init__(self, size, x=None, y=None):
        self.size = size
        if x is None:
            self.x = np.random.randint(0, size)
        else:
            self.x = x
        if y is None:
            self.y = np.random.randint(0, size)
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

    # COORDINATES ARE IMAGE BASED. I.e:
    # (x=SIZE,y=SIZE)=bottom rigth
    # and
    # (x=0, y=0)=top left
    def action(self, choice):
        if choice == UP:
            # move up 
            self.move(x=0, y=-1)
        elif choice == DOWN:
            #  move down
            self.move(x=0, y=1)
        elif choice == LEFT:
            # move left
            self.move(x=-1, y=0)
        elif choice == RIGHT:
            # move right
            self.move(x=1, y=0)
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
        if self.x < 0:
            self.x = 0
        elif self.x > self.size-1:
            self.x = self.size-1
        if self.y < 0:
            self.y = 0
        elif self.y > self.size-1:
            self.y = self.size-1
