from pool_simulation.physics import Simulation
from enum import Enum
from agent import Agent
import numpy as np

class TurnState(Enum):
    NORMAL = 0
    BALL_IN_HAND = 2
    BALL_IN_HAND_BAULK = 3
    GAME_OVER = 4


class Match:

    def __init__(self, engine: Simulation, play_break=False):
        self.engine = engine
        self.turn = 0
        self.open_table = True
        self.turn_state: TurnState = TurnState.NORMAL
        self.player_colours: dict[int, int | None] = {0: None, 1: None}
        self.winner = None
        if play_break:
            self.engine.reset_to_break()
            self.turn_state = TurnState.BALL_IN_HAND_BAULK
        else:
            self.engine.set_up_randomly(engine.n_obj_balls)

    def is_legal_shot(self, shot_data: dict):
        first_hit = shot_data["first_ball_hit"]
        potted = shot_data["balls_potted"]
        cushion = shot_data["cushion_after_impact"]

        is_foul = False

        if first_hit is None:
            is_foul = True  # Hit nothing
        elif 0 in potted:
            is_foul = True  # Potted the cue ball
        else:
            hit_color = self.engine.colours[first_hit]
            if self.open_table:
                # Cannot hit the Black first on an open table
                if hit_color == 3:
                    is_foul = True
            else:
                # Must hit your own color (or the black if you are on it)
                expected_color = 3 if self.is_on_black(self.turn) else self.player_colours[self.turn]
                if hit_color != expected_color:
                    is_foul = True

        # Rule 6p: Must hit a cushion after impact if nothing drops
        if len(potted) == 0 and not cushion:
            is_foul = True

        return not is_foul

    def is_on_black(self, turn):
        """Checks if the player has cleared all their designated colored balls."""
        if self.open_table:
            return False

        color = self.player_colours[turn]
        for i in range(1, self.engine.n_obj_balls + 1):
            if self.engine.in_play[i] and self.engine.colours[i] == color:
                return False  # Still have balls left!
        return True

    def play_turn(self, agent: Agent):

        if self.turn_state == TurnState.GAME_OVER:
            return

        if self.turn_state in [TurnState.BALL_IN_HAND_BAULK, TurnState.BALL_IN_HAND]:
            valid_placement = False
            while not valid_placement:
                pos = agent.get_cue_ball_in_hand_position(
                    self.engine.colours,
                    self.engine.in_play,
                    self.engine.positions
                )
                try:
                    # engine.move_cue_ball raises a ValueError if the spot overlaps another ball
                    self.engine.move_cue_ball(pos, baulk=self.turn_state==TurnState.BALL_IN_HAND_BAULK)
                    valid_placement = True
                except ValueError:
                    # In a real training scenario, penalize the AI here for an invalid move
                    pass

            self.turn_state = TurnState.NORMAL

        vel_x, vel_y, tip_x, tip_y, cue_elev = agent.get_shot_parameters(
            self.engine.colours,
            self.engine.in_play,
            self.engine.positions
        )

        self.engine.strike_cue_ball(vel_x, vel_y, tip_x, tip_y, cue_elev)

        shot_data = self.engine.run()

        # 4. Referee Evaluates the Result
        self.evaluate_shot(shot_data)

    def evaluate_shot(self, shot_data: dict):
        legal = self.is_legal_shot(shot_data)
        first_hit = shot_data["first_ball_hit"]
        potted = shot_data["balls_potted"]

        if shot_data["error"]:
            print("Engine Glitch Caught: Cue ball left the playing surface!")
            legal = False

        if 3 in potted:
            self.turn_state = TurnState.GAME_OVER
            if not legal or not self.is_on_black(self.turn):
                self.winner = 1 - self.turn
            else:
                self.winner = self.turn
            return

        if not legal:
            self.turn = 1 - self.turn  # Switch turns
            self.turn_state = TurnState.BALL_IN_HAND

            if not self.engine.in_play[0]:
                self.engine.positions[0] = np.array([0.0, -0.7])
                self.engine.in_play[0] = True
            return

        self.turn_state = TurnState.NORMAL

        # A. Deciding Groups (Rule 6a)
        if self.open_table and len(potted) > 0:
            first_potted_idx = potted[0]
            first_potted_colour = int(self.engine.colours[first_potted_idx])
            first_hit_colour = int(self.engine.colours[first_hit])

            if first_potted_colour in [1, 2] and first_hit_colour == first_potted_colour:
                self.player_colours[self.turn] = first_potted_colour
                self.player_colours[1 - self.turn] = 2 if first_potted_colour == 1 else 1
                self.open_table = False
                return

        # B. Continuing the visit (Rule 6b)
        if not self.open_table:
            my_color = self.player_colours[self.turn]
            potted_colors = [self.engine.colours[i] for i in potted]

            if my_color in potted_colors:
                return  # Keeps turn

        # C. Legal shot, but no pot
        self.turn = 1 - self.turn