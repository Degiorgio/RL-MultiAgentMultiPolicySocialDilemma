import arcade
from gridworld import BlobEnv
from blob import UP, DOWN, LEFT, RIGHT, NOTHING

GRID_WORLD_SIZE = 20
WIDTH = GRID_WORLD_SIZE
HEIGHT = GRID_WORLD_SIZE
MARGIN = 1
SCREEN_WIDTH = (WIDTH + MARGIN) * GRID_WORLD_SIZE + MARGIN
SCREEN_HEIGHT = (HEIGHT + MARGIN) * GRID_WORLD_SIZE + MARGIN


class Gathering(arcade.Window):
    def __init__(self):
        self.env = BlobEnv(size=20)
        self.set_update_rate(1 / 10)
        self.total_reward = 0
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT)

    def setup(self):
        self.env.reset(num_players=2)
        print(self.env.players)

    def update(self, dt):
        return

    def on_key_press(self, key, modifiers):
        """
        called whenever the kep is pressed.
        """
        action = NOTHING
        action2 = NOTHING
        if key == arcade.key.UP:
            action = UP
        elif key == arcade.key.DOWN:
            action = DOWN
        elif key == arcade.key.RIGHT:
            action = RIGHT
        elif key == arcade.key.LEFT:
            action = LEFT
        elif key == arcade.key.W:
            action2 = UP
        elif key == arcade.key.S:
            action2 = DOWN
        elif key == arcade.key.D:
            action2 = RIGHT
        elif key == arcade.key.A:
            action2 = LEFT
        elif key == arcade.key.C:
            action = NOTHING
        elif key == arcade.key.ENTER:
            self.env.render()
        if action is not None:
            state, reward, done = self.env.step([action, action2])
            print(f"Player locations {self.env.players}")
            print(f"Action rewards {reward}")

    def on_draw(self):
        """
        Render the screen.
        """
        arcade.start_render()
        for row in range(GRID_WORLD_SIZE):
            for column in range(GRID_WORLD_SIZE):
                # figure out what color to draw the box
                if self.env.contains_food(row, column):
                    color = arcade.color.GREEN
                elif self.env.contains_player(row, column, 1):
                    color = arcade.color.BLUE
                elif self.env.contains_player(row, column, 2):
                    color = arcade.color.RED
                else:
                    color = arcade.color.BLACK
                x = (MARGIN + WIDTH) * column + MARGIN + WIDTH // 2
                y = (MARGIN + HEIGHT) * row + MARGIN + HEIGHT // 2
                arcade.draw_rectangle_filled(x, y, WIDTH, HEIGHT, color)

def main():
    window = Gathering()
    window.setup()
    arcade.run()


if __name__ == "__main__":
    main()
