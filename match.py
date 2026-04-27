from pool_simulation.constants import *
from pool_simulation.physics import Simulation
from enum import IntEnum
import numpy as np


class TurnState(IntEnum):
    NORMAL = 0
    BALL_IN_HAND = 1
    BALL_IN_HAND_BAULK = 2
    GAME_OVER = 3


class Match:
    def __init__(self, engine: Simulation, play_break=False, custom_setup = True):
        self.engine = engine
        self.turn = 0
        self.open_table = True
        self.turn_state: TurnState = TurnState.NORMAL
        self.player_colours: dict[int, int | None] = {0: None, 1: None}
        self.winner = None

        self.is_break_shot = play_break

        if play_break:
            self.engine.reset_to_break()
            self.turn_state = TurnState.BALL_IN_HAND_BAULK
        elif not custom_setup:
            self.engine.set_up_randomly(engine.n_obj_balls)

    def was_on_black(self, turn, potted_this_shot=None):
        """Checks if the player was on the black before the shot occurred."""
        if self.open_table:
            return False

        if potted_this_shot is None:
            potted_this_shot = []

        color = self.player_colours[turn]
        for i in range(1, self.engine.n_obj_balls + 1):
            # If the ball is still on the table OR was just potted, the player wasn't on the black!
            if (self.engine.in_play[i] or i in potted_this_shot) and self.engine.colours[i] == color:
                return False
        return True

    def play_turn(self, agent, frame_callback=None):
        if self.turn_state == TurnState.GAME_OVER:
            return self.winner

        # if self.is_break_shot:
        #     print(f"Player {self.turn + 1} to break.")
        # else:
        #     if self.open_table:
        #         print(f"Player {self.turn + 1} to play. Open Table.")
        #     else:
        #         print(f"Player {self.turn + 1}: {COLOUR_NAMES[self.player_colours[self.turn]]} Balls in play.")

        # 1. Handle Ball in Hand placements
        if self.turn_state in [TurnState.BALL_IN_HAND_BAULK, TurnState.BALL_IN_HAND]:
            valid_placement = False
            while not valid_placement:
                pos = agent.get_cue_ball_in_hand_position(
                    self.engine.colours,
                    self.engine.in_play,
                    self.engine.positions,
                    self.player_colours[self.turn],
                    self.turn_state
                )
                try:
                    self.engine.move_cue_ball(pos, baulk=(self.turn_state == TurnState.BALL_IN_HAND_BAULK))
                    valid_placement = True
                except ValueError:
                    self.turn_state = TurnState.BALL_IN_HAND
                    self.turn = 1 - self.turn
                    return None

            self.turn_state = TurnState.NORMAL
        # 2. Get and execute shot
        vel_x, vel_y, tip_y, tip_x, cue_elev = agent.get_shot_parameters(
            self.engine.colours,
            self.engine.in_play,
            self.engine.positions,
            self.player_colours[self.turn],
            self.turn_state
        )

        valid = self.engine.strike_cue_ball(vel_x, vel_y, tip_y, tip_x, cue_elev)
        shot_data = self.engine.run(framerate=FPS, frame_callback=frame_callback)
        shot_data["valid"] = valid
        # 3. Referee evaluates the result
        self.evaluate_shot(shot_data)
        return vel_x, vel_y, tip_y, tip_x, cue_elev

    def respot_ball(self, ball_idx):
        """Places a ball on the black spot, or slides it towards the top cushion if blocked."""
        # Start at the exact Black Spot
        candidate_pos = np.array([BLACK_SPOT_X, 0.0])

        # Safe distance is the sum of the two radii plus a microscopic gap so they don't trigger a collision
        my_radius = self.engine.radii[ball_idx]

        resolved = False
        while not resolved:
            resolved = True

            for i in range(self.engine.n_obj_balls + 1):
                # Skip ourselves and balls that are already off the table
                if i == ball_idx or not self.engine.in_play[i]:
                    continue

                other_pos = self.engine.positions[i]
                other_radius = self.engine.radii[i]
                safe_dist = my_radius + other_radius + 1e-5  # 1e-5 is the "without touching" gap

                # Check distance between our candidate spot and this ball
                dist = np.linalg.norm(other_pos - candidate_pos)

                if dist < safe_dist:
                    # WE HAVE AN OVERLAP!
                    resolved = False

                    # Calculate exactly how far forward along the X-axis we need to slide
                    # to clear this specific ball perfectly.
                    dy = other_pos[1] - candidate_pos[1]  # Distance on the Y axis

                    # Pythagorean theorem: dx^2 + dy^2 = safe_dist^2
                    dx = np.sqrt(safe_dist ** 2 - dy ** 2)

                    # Push the candidate position forward past the obstructing ball
                    candidate_pos[0] = other_pos[0] + dx

                    # Break the inner loop and re-verify this new spot against ALL balls again
                    break

        # Once the while loop finishes, candidate_pos is guaranteed to be safe and legal!
        self.engine.positions[ball_idx] = candidate_pos
        self.engine.in_play[ball_idx] = True
        self.engine.velocities[ball_idx].fill(0.0)
        self.engine.angular[ball_idx].fill(0.0)
        self.engine.ball_states[ball_idx] = "STOPPED"
        self.engine.ball_versions[ball_idx] += 1

    def evaluate_shot(self, shot_data: dict):
        first_hit = shot_data.get("first_ball_hit")
        potted = shot_data.get("balls_potted", [])
        cushion = shot_data.get("cushion_after_ball", False)
        error_balls = shot_data.get("error_balls", [])  # List of ball indices that left the table
        balls_past_middle = shot_data.get("balls_past_middle", set())  # Set of balls that crossed the line
        valid = shot_data.get("valid", True)

        # ==========================================
        # 1. HANDLE BALLS OFF THE TABLE (Rules 6l & 6m)
        # ==========================================
        for ball_idx in error_balls:
            if ball_idx != 0:
                self.respot_ball(ball_idx)

        cue_ball_off_table = 0 in error_balls

        # ==========================================
        # 2. THE BREAK SHOT (Rule 4)
        # ==========================================
        if self.is_break_shot:
            self.is_break_shot = False  # Table remains open after break regardless

            # Do not count the cue ball as a point!
            obj_potted = [b for b in potted if b != 0]

            # Rule 4f: The 3-Point Rule
            break_points = len(obj_potted) + len(balls_past_middle)

            if break_points < 3 or not valid:
                # print("Failure to perform legal break. Re-rack required.")
                self.turn = 1 - self.turn
                # In a real game, opponent chooses to break or pass back. Here we just switch turns and re-rack.
                self.engine.reset_to_break()
                self.turn_state = TurnState.BALL_IN_HAND_BAULK
                self.is_break_shot = True
                return

            # Rule 4i: 8-ball potted on break is re-spotted, NOT loss of frame
            if 3 in [self.engine.colours[i] for i in obj_potted]:
                # print("8-Ball Potted on Break, Re-Spot.")
                for idx in obj_potted:
                    if self.engine.colours[idx] == 3:
                        self.respot_ball(idx)
                        potted.remove(idx)

            # Rule 4j: Fouls on the break
            if 0 in potted:  # In-off
                # print("Cue Ball Potted on break: Ball in hand from baulk")
                self.turn = 1 - self.turn
                self.turn_state = TurnState.BALL_IN_HAND_BAULK
                return
            elif cue_ball_off_table:
                # print("Cue Ball glitched off table: Ball in hand")
                self.turn = 1 - self.turn
                self.turn_state = TurnState.BALL_IN_HAND
                return
            elif first_hit is None or self.engine.colours[first_hit] == 3:
                # print("Standard foul on the break: Ball in hand")
                self.turn = 1 - self.turn
                self.turn_state = TurnState.BALL_IN_HAND
                return

            # Legal break. If nothing potted, pass turn.
            if len(obj_potted) == 0:
                # print("Legal Break")
                self.turn = 1 - self.turn

            return

        # ==========================================
        # 3. STANDARD PLAY FOUL DETECTION
        # ==========================================
        is_foul = False

        if first_hit is None or 0 in potted or cue_ball_off_table:
            is_foul = True
        else:
            hit_color = self.engine.colours[first_hit]
            if self.open_table:
                if hit_color == 3:
                    is_foul = True
            else:
                expected_color = 3 if self.was_on_black(self.turn, potted) else self.player_colours[self.turn]
                if hit_color != expected_color:
                    is_foul = True

        if (len(potted) == 0 and not cushion) or not valid:
            is_foul = True

        # ==========================================
        # 4. BLACK BALL LOGIC (Win/Loss)
        # ==========================================
        potted_colors = [int(self.engine.colours[i]) for i in potted]

        if 3 in potted_colors:
            self.turn_state = TurnState.GAME_OVER
            if is_foul or not self.was_on_black(self.turn, potted):
                self.winner = 1 - self.turn
            else:
                self.winner = self.turn
            # print("Black Ball Potted.")
            return

        # ==========================================
        # 5. HANDLE STANDARD FOULS
        # ==========================================
        if is_foul:
            # print("Foul")
            self.turn = 1 - self.turn
            self.turn_state = TurnState.BALL_IN_HAND

            if not self.engine.in_play[0]:
                self.engine.positions[0] = np.array([0.0, -0.7])
                self.engine.in_play[0] = True
            return

        self.turn_state = TurnState.NORMAL

        # ==========================================
        # 6. GROUPS AND CONTINUATION
        # ==========================================
        first_hit_colour = int(self.engine.colours[first_hit])

        # A. Deciding Groups (Rule 6a)
        if self.open_table and len(potted) > 0:
            # Rule 6a.4: If you strike a group first, and pot a ball of that group, you get that group
            # (even if another group fell into the pocket first chronologically)
            if first_hit_colour in [1, 2] and first_hit_colour in potted_colors:
                self.player_colours[self.turn] = first_hit_colour
                self.player_colours[1 - self.turn] = 2 if first_hit_colour == 1 else 1
                self.open_table = False
                return  # Keeps turn

        # B. Continuing the visit (Rule 6b)
        if not self.open_table:
            my_color = self.player_colours[self.turn]
            if my_color in potted_colors:
                return  # Keeps turn

        # C. Legal shot, but no pot / potted wrong color
        self.turn = 1 - self.turn