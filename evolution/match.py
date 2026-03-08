from pool_simulation.constants import BLACK_SPOT_X
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

        self.is_break_shot = play_break

        if play_break:
            self.engine.reset_to_break()
            self.turn_state = TurnState.BALL_IN_HAND_BAULK
        else:
            self.engine.set_up_randomly(engine.n_obj_balls)

    def is_on_black(self, turn):
        """Checks if the player has cleared all their designated colored balls."""
        if self.open_table:
            return False

        color = self.player_colours[turn]
        for i in range(1, self.engine.n_obj_balls + 1):
            if self.engine.in_play[i] and self.engine.colours[i] == color:
                return False
        return True

    def play_turn(self, agent: Agent):
        if self.turn_state == TurnState.GAME_OVER:
            return

        # 1. Handle Ball in Hand placements
        if self.turn_state in [TurnState.BALL_IN_HAND_BAULK, TurnState.BALL_IN_HAND]:
            valid_placement = False
            while not valid_placement:
                pos = agent.get_cue_ball_in_hand_position(
                    self.engine.colours,
                    self.engine.in_play,
                    self.engine.positions
                )
                try:
                    self.engine.move_cue_ball(pos, baulk=(self.turn_state == TurnState.BALL_IN_HAND_BAULK))
                    valid_placement = True
                except ValueError:
                    pass

            self.turn_state = TurnState.NORMAL

        # 2. Get and execute shot
        vel_x, vel_y, tip_x, tip_y, cue_elev = agent.get_shot_parameters(
            self.engine.colours,
            self.engine.in_play,
            self.engine.positions
        )

        self.engine.strike_cue_ball(vel_x, vel_y, tip_x, tip_y, cue_elev)
        shot_data = self.engine.run()

        # 3. Referee evaluates the result
        self.evaluate_shot(shot_data)

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
        cushion = shot_data.get("cushion_after_impact", False)
        error_balls = shot_data.get("error_balls", [])  # List of ball indices that left the table
        balls_past_centre = shot_data.get("balls_past_centre", set())  # Set of balls that crossed the line

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

            # Rule 4i: 8-ball potted on break is re-spotted, NOT loss of frame
            if 3 in [self.engine.colours[i] for i in potted]:
                for idx in potted:
                    if self.engine.colours[idx] == 3:
                        self.respot_ball(idx)
                        potted.remove(idx)

            # Rule 4j: Fouls on the break
            if 0 in potted:  # In-off
                self.turn = 1 - self.turn
                self.turn_state = TurnState.BALL_IN_HAND_BAULK
                return
            elif cue_ball_off_table:
                self.turn = 1 - self.turn
                self.turn_state = TurnState.BALL_IN_HAND
                return

            # Rule 4f: The 3-Point Rule
            break_points = len(potted) + len(balls_past_centre)
            if break_points < 3:
                print("Failure to perform legal break. Re-rack required.")
                self.turn = 1 - self.turn
                # In a real game, opponent chooses to break or pass back. Here we just switch turns and re-rack.
                self.engine.reset_to_break()
                self.turn_state = TurnState.BALL_IN_HAND_BAULK
                self.is_break_shot = True
                return

            # Legal break. If nothing potted, pass turn.
            if len(potted) == 0:
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
                expected_color = 3 if self.is_on_black(self.turn) else self.player_colours[self.turn]
                if hit_color != expected_color:
                    is_foul = True

        if len(potted) == 0 and not cushion:
            is_foul = True

        # ==========================================
        # 4. BLACK BALL LOGIC (Win/Loss)
        # ==========================================
        potted_colors = [int(self.engine.colours[i]) for i in potted]

        if 3 in potted_colors:
            self.turn_state = TurnState.GAME_OVER
            if is_foul or not self.is_on_black(self.turn):
                self.winner = 1 - self.turn
            else:
                self.winner = self.turn
            return

        # ==========================================
        # 5. HANDLE STANDARD FOULS
        # ==========================================
        if is_foul:
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