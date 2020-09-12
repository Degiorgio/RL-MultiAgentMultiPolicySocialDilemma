from env.gridworld import FOOD_TINY, FOOD_LITTLE, FOOD_NORMAL, FOOD_ALOT

GRID_WORLD_SIZE = 42

def env_scarse(murder_mode=True,
               steps_per_episode=1000,
               beam_color_diff=True,
               draw_beam=True,
               draw_shooting_direction=True,
               shoot_in_all_directions=True,
               food_respawn_time=14,
               small_world=True,
               player_move_cost=0,
               player_respawn_time=200):
    return {
        "size": GRID_WORLD_SIZE,
        "num_players": 2,
        "murder_mode": murder_mode,
        "steps_per_episode": steps_per_episode,
        "player_move_cost": player_move_cost,
        "player_respawn_time": player_respawn_time,
        "food_reward": 50,
        "food_respawn_time": food_respawn_time,
        "food_level": FOOD_LITTLE,
        "beam_color_diff": beam_color_diff,
        "draw_beam": draw_beam,
        "draw_shooting_direction": draw_shooting_direction,
        "shoot_in_all_directions": shoot_in_all_directions,
        "small_world":  small_world
    }


def env_very_scarse(murder_mode=True, 
                    steps_per_episode=1000, 
                    beam_color_diff=True, 
                    draw_beam=True, 
                    draw_shooting_direction=True, 
                    shoot_in_all_directions=True, 
                    small_world=True):
    return {
            "size": GRID_WORLD_SIZE,
            "num_players": 2,
            "murder_mode": murder_mode,
            "steps_per_episode": steps_per_episode,
            "player_move_cost": 0,
            "player_respawn_time": 200,
            "food_reward": 50,
            "food_respawn_time": 5,
            "food_level": FOOD_TINY,
            "beam_color_diff": beam_color_diff,
            "draw_beam": draw_beam,
            "draw_shooting_direction": draw_shooting_direction,
            "shoot_in_all_directions": shoot_in_all_directions,
            "small_world":  small_world
    }


def env_abudent(food_respawn_time,
                player_respawn_time,
                murder_mode=True,
                steps_per_episode=1000,
                beam_color_diff=True,
                draw_beam=True,
                draw_shooting_direction=True,
                shoot_in_all_directions=True,
                small_world=True):
    return {
            "size": GRID_WORLD_SIZE,
            "num_players": 2,
            "murder_mode": murder_mode,
            "steps_per_episode": steps_per_episode,
            "player_move_cost": 0,
            "player_respawn_time": player_respawn_time,
            "food_reward": 50,
            "food_respawn_time": food_respawn_time,
            "food_level": FOOD_ALOT,
            "beam_color_diff": beam_color_diff,
            "draw_beam": draw_beam,
            "draw_shooting_direction": draw_shooting_direction,
            "shoot_in_all_directions": shoot_in_all_directions,
            "small_world":  small_world
    }


def env_normal(murder_mode=True,
               steps_per_episode=1000,
               beam_color_diff=True,
               draw_beam=True,
               draw_shooting_direction=True,
               shoot_in_all_directions=True,
               small_world=True,
               player_move_cost=0,
               player_respawn_time=200):
    return {
            "size": GRID_WORLD_SIZE,
            "num_players": 2,
            "murder_mode": murder_mode,
            "steps_per_episode": steps_per_episode,
            "player_move_cost": player_move_cost,
            "player_respawn_time": player_respawn_time,
            "food_reward": 50,
            "food_respawn_time": 40,
            "food_level": FOOD_NORMAL,
            "beam_color_diff": beam_color_diff,
            "draw_beam": draw_beam,
            "draw_shooting_direction": draw_shooting_direction,
            "shoot_in_all_directions": shoot_in_all_directions,
            "small_world":  small_world
    }