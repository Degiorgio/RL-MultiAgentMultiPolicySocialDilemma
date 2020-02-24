import arcade
from env.gridworld import GatheringEnv

# actions
from env.blob import UP, DOWN, LEFT, RIGHT, NOTHING
from env.blob import ROTATE_LEFT, ROTATE_RIGHT, SHOOT

GRID_WORLD_SIZE = 20
WIDTH = GRID_WORLD_SIZE
HEIGHT = GRID_WORLD_SIZE
MARGIN = 1
SCREEN_WIDTH = (WIDTH + MARGIN) * GRID_WORLD_SIZE + MARGIN
SCREEN_HEIGHT = (HEIGHT + MARGIN) * GRID_WORLD_SIZE + MARGIN

NUM_PLAYERS = 2


class Gathering(arcade.Window):
    def __init__(self):
        if NUM_PLAYERS == 1:
            self.env = GatheringEnv(size=20,
                                    num_players=NUM_PLAYERS,
                                    player_murder_mode=False)
        else:
            self.env = GatheringEnv(size=20,
                                    num_players=NUM_PLAYERS,
                                    player_move_cost=True)
        self.set_update_rate(1 / 10)
        self.acted = None
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT)

    def setup(self):
        self.env.reset()
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
        elif key == arcade.key.Q:
            action = ROTATE_LEFT
        elif key == arcade.key.E:
            action = ROTATE_RIGHT
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
        elif key == arcade.key.BACKSPACE:
            self.env.render(close=True)
        elif key == arcade.key.SPACE:
            action = SHOOT
        elif key == arcade.key.HOME:
            action2 = SHOOT
        if action is not None:
            try:
                if NUM_PLAYERS == 2:
                    states, rewards, done, info = self.env.step([action, action2])
                else:
                    states, rewards, done, info = self.env.step([action])
            except Exception as e:
                import traceback
                traceback.print_exception(e)
                import ipdb; ipdb.set_trace()

            print(f"Player locations {self.env.players}")
            print(f"Action rewards {rewards}")
        self.acted = True

    def on_draw(self):
        """
        Render the screen.
        """
        arcade.start_render()
        for row in range(GRID_WORLD_SIZE):
            for column in range(GRID_WORLD_SIZE):
                x = (MARGIN + WIDTH) * column + MARGIN + WIDTH // 2
                y = (MARGIN + HEIGHT) * row + MARGIN + HEIGHT // 2
                # figure out what color to draw the box
                if self.env._contains_food(row, column):
                    color = arcade.color.GREEN
                elif self.env._contains_player(row, column, 1):
                    color = arcade.color.BLUE
                elif self.env._contains_player(row, column, 2):
                    color = arcade.color.RED
                elif self.env._contains_player_direction(row, column):
                    color = arcade.color.GRAY
                else:
                    color = arcade.color.BLACK
                arcade.draw_rectangle_filled(x, y, WIDTH, HEIGHT, color)


def main():
    window = Gathering()
    window.setup()
    arcade.run()


if __name__ == "__main__":
    main()
