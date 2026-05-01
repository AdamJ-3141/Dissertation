import json

import numpy as np
from pool_simulation.constants import *
from planner.evaluator import TableEvaluator
from planner.shot_planner import ShotPlanner
from match import TurnState


class Agent:
    def __init__(self, sim, weights):
        self.sim = sim
        self.weights = weights

    def get_shot_parameters(self, colours: np.ndarray, in_play: np.ndarray, positions: np.ndarray,
                            target_color: int, turn_state: int, renderer=None) -> tuple:

        # Initialize Evaluator and Planner with current Agent weights
        # self.weights should be a dict of the customizable parameters we defined
        evaluator = TableEvaluator(self.sim, target_color, weights=self.weights)
        planner = ShotPlanner(self.sim, evaluator, self.weights)

        # Ask the Planner for the best strategic decision
        # Returns ("offensive"|"safety", (aim_angle, power, topspin, sidespin, elevation))
        decision_type, params = planner.find_best_shot(renderer=renderer)

        if params is None:
            # Fallback: Tap the cue ball if totally lost, shouldn't happen too often
            return 0.05, 0.0, 0.0, 0.0, 0.0

        aim_angle, power, topspin, sidespin, elevation = params

        # Convert polar shot power and angle to Cartesian velocity
        vx = power * np.cos(aim_angle)
        vy = power * np.sin(aim_angle)

        return vx, vy, topspin, sidespin, elevation

    def get_cue_ball_in_hand_position(self, colours, in_play, positions, target_color, turn_state):
        evaluator = TableEvaluator(self.sim, target_color, weights=self.weights)

        # Identify valid targets (Own colour, or Black if cleared)
        if target_color is None:
            my_targets = [i for i in range(1, self.sim.n_obj_balls + 1)
                          if colours[i] in (1, 2) and in_play[i]]
        else:
            my_targets = [i for i in range(1, self.sim.n_obj_balls + 1)
                          if colours[i] == target_color and in_play[i]]

            if not my_targets:
                my_targets = [i for i in range(1, self.sim.n_obj_balls + 1) if colours[i] == 3 and in_play[i]]

        # Prioritize Targets
        target_scores = []
        for t_idx in my_targets:
            t_pos = positions[t_idx][:2]

            # Count visible pockets
            visible_pockets = sum(1 for p_idx in range(len(self.sim.pockets))
                                  if evaluator.get_pocket_access_multiplier(t_idx, p_idx) > 0)

            # Count how many balls are physically touching/blocking it (Congestion)
            congestion = sum(1 for j in range(1, self.sim.n_obj_balls + 1)
                             if in_play[j] and j != t_idx and np.linalg.norm(t_pos - positions[j][:2]) < 0.15)

            # Lower score = Harder ball (High congestion, few pockets)
            priority_score = visible_pockets - (congestion * 2)
            target_scores.append((priority_score, t_idx))

        # Sort so the hardest balls are targeted first
        target_scores.sort(key=lambda x: x[0])

        # # Calculate Perfect Placement
        # baulk_limit_x = (-TABLE_WIDTH / 2) + BAULK_LINE_X if turn_state == TurnState.BALL_IN_HAND_BAULK else None
        #
        # for _, t_idx in target_scores:
        #     t_pos = positions[t_idx][:2]
        #
        #     for p_idx in range(len(self.sim.pockets)):
        #         if evaluator.get_pocket_access_multiplier(t_idx, p_idx) > 0:
        #             p_pos = self.sim.pockets[p_idx][:2]
        #             dir_tp = (p_pos - t_pos) / np.linalg.norm(p_pos - t_pos)
        #             gb_pos = t_pos - (2 * OBJECT_BALL_RADIUS) * dir_tp
        #
        #             # Ideal placement: A perfect straight-in shot, 30cm behind the ghost ball
        #             ideal_cb_pos = gb_pos - (dir_tp * 0.3)
        #
        #             # A. Check Baulk restriction
        #             if baulk_limit_x is not None and ideal_cb_pos[0] > baulk_limit_x:
        #                 continue
        #
        #             # B. Check table bounds (ensure it isn't placed on the cushion)
        #             if not (-TABLE_WIDTH / 2 + CUE_BALL_RADIUS < ideal_cb_pos[0] < TABLE_WIDTH / 2 - CUE_BALL_RADIUS and
        #                     -TABLE_HEIGHT / 2 + CUE_BALL_RADIUS < ideal_cb_pos[1] < TABLE_HEIGHT / 2 - CUE_BALL_RADIUS):
        #                 continue
        #
        #             # C. Check overlap with other physical balls
        #             overlap = any(in_play[j] and np.linalg.norm(ideal_cb_pos - positions[j][:2]) < 2 * CUE_BALL_RADIUS
        #                           for j in range(1, self.sim.n_obj_balls + 1))
        #             if overlap: continue
        #
        #             # D. Check if the path from this perfect spot to the ghost ball is actually clear
        #             if evaluator.is_path_clear(ideal_cb_pos, gb_pos, ignore_indices=[t_idx]):
        #                 return ideal_cb_pos

        # Fallback: Heatmap Peak with Pot Clearance
        tactical_map, w, h, _ = evaluator.get_full_heatmap()
        ny, nx = tactical_map.shape
        x_coords = np.linspace(-w, w, nx)
        y_coords = np.linspace(-h, h, ny)

        best_pos = np.array([0.0, 0.0])
        max_val = -np.inf

        # Check if we are restricted to the baulk area
        baulk_limit_x = (-TABLE_WIDTH / 2) + BAULK_LINE_X if turn_state == TurnState.BALL_IN_HAND_BAULK else None

        for i in range(ny):
            for j in range(nx):
                current_val = tactical_map[i, j]
                # Skip immediately if this pixel is worse than our current best
                if current_val <= max_val: continue

                test_pos = np.array([x_coords[j], y_coords[i]])

                # A. Check Baulk restriction
                if baulk_limit_x is not None and test_pos[0] > baulk_limit_x:
                    continue

                # B. Check physical overlap with other balls
                overlap = any(in_play[b] and np.linalg.norm(test_pos - positions[b][:2]) < 2 * CUE_BALL_RADIUS
                              for b in range(1, self.sim.n_obj_balls + 1))
                if overlap: continue

                # C. Check if this spot actually has a clear path to pot something
                can_pot = False
                for t_idx in my_targets:
                    for p_idx in range(len(self.sim.pockets)):
                        if evaluator.get_pocket_access_multiplier(t_idx, p_idx) > 0:
                            p_pos = self.sim.pockets[p_idx][:2]
                            dir_tp = (p_pos - positions[t_idx][:2]) / np.linalg.norm(
                                p_pos - positions[t_idx][:2])
                            gb_pos = positions[t_idx][:2] - (2 * OBJECT_BALL_RADIUS) * dir_tp

                            if evaluator.is_path_clear(test_pos, gb_pos, ignore_indices=[t_idx]):
                                can_pot = True
                                break
                    if can_pot: break

                if can_pot:
                    max_val = current_val
                    best_pos = test_pos

        # Absolute Desperation Fallback (If no pots are physically possible from anywhere)
        if max_val == -np.inf:
            for i in range(ny):
                for j in range(nx):
                    test_pos = np.array([x_coords[j], y_coords[i]])

                    # Check Baulk restriction
                    if baulk_limit_x is not None and test_pos[0] > baulk_limit_x:
                        continue

                    # Check physical overlap
                    overlap = any(in_play[b] and np.linalg.norm(test_pos - positions[b][:2]) < 2 * CUE_BALL_RADIUS
                                  for b in range(1, self.sim.n_obj_balls + 1))

                    if not overlap:
                        return test_pos

        return best_pos


class Human(Agent):
    def __init__(self, sim,  renderer):
        super().__init__(sim, weights=None)
        self.renderer = renderer

        # UI State
        self.tip_x = 0.0
        self.tip_y = 0.0
        self.power = 4.0  # Base power in m/s
        self.elevation = 0.0
        self.spin_step = 0.05
        self.power_step = 0.1
        self.elevation_step = 2

    def get_cue_ball_in_hand_position(self, colours, in_play, positions,
                                      target_colour, turn_state, renderer=None) -> np.ndarray | None:
        """Pygame loop to let the user drag the cue ball."""
        placed = False
        import pygame
        while not placed:
            self.renderer.render(fps=60, flip=False)

            # Draw a glowing cue ball at the mouse cursor
            mouse_pos = pygame.mouse.get_pos()
            pygame.draw.circle(self.renderer.screen, (200, 255, 200), mouse_pos,
                               int(OBJECT_BALL_RADIUS * self.renderer.scale))

            pygame.display.flip()
            self.renderer.clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # User clicked! Convert screen coords back to world coords
                    world_pos = self.renderer.screen_to_world(event.pos)
                    return np.array(world_pos)
        return None

    def get_shot_parameters(self, colours, in_play, positions, target_colour, turn_state, renderer=None) -> tuple:
        """Pygame loop to let the user aim and set spin."""
        shot_locked = False
        import pygame
        # Clear any leftover clicks from the ball-in-hand phase
        pygame.event.clear()

        self.tip_x = 0.0
        self.tip_y = 0.0
        self.power = 8.0 if self.sim.is_break else 2.0 # Base power in m/s
        self.elevation = 0.0  # This now acts as the player's "Manual Override"

        vel_x = 0.0
        vel_y = 0.0
        actual_elevation = 0.0

        while not shot_locked:
            self.renderer.render(fps=60, flip=False)

            # Handle Aiming (Mouse)
            mouse_pos = pygame.mouse.get_pos()
            target_world_pos = self.renderer.screen_to_world(mouse_pos)
            cue_world_pos = positions[0]
            direction = target_world_pos - cue_world_pos
            dist = np.linalg.norm(direction)

            # Calculate the theoretical velocities based on the mouse position
            test_vx, test_vy = 0.0, 0.0
            if dist > 1e-4:
                v_hat = direction / dist
                test_vx = float(v_hat[0] * self.power)
                test_vy = float(v_hat[1] * self.power)

            min_el = 89.0  # Fallback to maximum elevation
            if dist > 1e-4:
                # Ask the engine to find the lowest valid angle for this exact shot
                for test_e in range(0, 90):
                    if self.sim.validate_shot(test_vx, test_vy, self.tip_y, self.tip_x, float(test_e)):
                        min_el = float(test_e)
                        break

            # The final elevation is whichever is higher: the required physical minimum, or the player's manual input
            actual_elevation = max(self.elevation, min_el)

            # Double check that the final calculated shot is legal
            is_valid = False
            if dist > 1e-4:
                is_valid = self.sim.validate_shot(test_vx, test_vy, self.tip_y, self.tip_x, actual_elevation)

            # Only draw the aiming line if the shot is physically possible
            if is_valid:
                self.renderer.draw_aim_line(mouse_pos[0], mouse_pos[1], self.power, self.tip_y, self.tip_x,
                                            actual_elevation)

            self.renderer.draw_power_scale(self.power)
            self.renderer.draw_spin_ui(self.tip_x, self.tip_y)

            # Pass our dynamically calculated elevation to the UI so the player sees the cue raise!
            self.renderer.draw_elevation_ui(actual_elevation)

            pygame.display.flip()
            self.renderer.clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

                # SPIN CONTROLS (Arrow Keys)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.tip_y += self.spin_step
                    elif event.key == pygame.K_DOWN:
                        self.tip_y -= self.spin_step
                    elif event.key == pygame.K_RIGHT:
                        self.tip_x += self.spin_step
                    elif event.key == pygame.K_LEFT:
                        self.tip_x -= self.spin_step

                    # POWER CONTROLS (W / S)
                    elif event.key == pygame.K_w:
                        self.power = min(MAX_SPEED, self.power + self.power_step)
                    elif event.key == pygame.K_s:
                        self.power = max(0.2, self.power - self.power_step)

                    # Clamp spin to a physically realistic circle
                    norm = float(np.linalg.norm([self.tip_x, self.tip_y]))
                    if norm > 0.8:
                        self.tip_x = (self.tip_x / norm) * 0.8
                        self.tip_y = (self.tip_y / norm) * 0.8

                    # ELEVATION CONTROLS (E / D) - This now acts as a manual floor
                    elif event.key == pygame.K_e:
                        self.elevation = min(89.0, self.elevation + self.elevation_step)
                    elif event.key == pygame.K_d:
                        self.elevation = max(0.0, self.elevation - self.elevation_step)

                # POWER CONTROLS (Scroll Wheel)
                elif event.type == pygame.MOUSEWHEEL:
                    self.power = np.clip(self.power + (event.y * self.power_step), 0.1, MAX_SPEED)

                # FIRE SHOT (Click)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # ONLY allow the shot to fire if it cleared the validation check!
                    if is_valid:
                        vel_x = test_vx
                        vel_y = test_vy
                        shot_locked = True

        return vel_x, vel_y, self.tip_y, self.tip_x, actual_elevation


class RandomAgent(Agent):

    def __init__(self, sim):
        super().__init__(sim, None)
        self.sim = sim

    def get_shot_parameters(self, colours: np.ndarray, in_play: np.ndarray, positions: np.ndarray,
                            target_color: int, turn_state: int, renderer=None) -> tuple:
        import random

        vx = random.uniform(0.2, 4.0)
        vy = random.uniform(0.2, 4.0)
        topspin = random.uniform(-0.75, 0.75)
        sidespin = random.uniform(-0.75, 0.75)

        min_el = 89.0  # Fallback to maximum elevation
        for test_e in range(0, 90):
            if self.sim.validate_shot(vx, vy, topspin, sidespin, float(test_e)):
                min_el = float(test_e)
                break

        return vx, vy, topspin, sidespin, min_el

    def get_cue_ball_in_hand_position(self, colours, in_play, positions, target_color, turn_state):
        import random

        valid = False
        pos_x, pos_y = 0, 0
        while not valid:
            pos_x = random.uniform(-TABLE_WIDTH / 2 + CUSHION_WIDTH, TABLE_WIDTH / 2 - CUSHION_WIDTH)
            pos_y = random.uniform(-TABLE_HEIGHT / 2 + CUSHION_WIDTH, TABLE_HEIGHT / 2 - CUSHION_WIDTH)
            valid = True
            for p in positions:
                if (p[0] - pos_x)**2 + (p[1] - pos_y)**2 < (2*OBJECT_BALL_RADIUS)**2:
                    valid = False
        return np.array([pos_x, pos_y])


class GreedyAgent(Agent):

    def __init__(self, sim):
        with open("planner/greedy_defaults.json") as f:
            w = json.load(f)
        super().__init__(sim, w)
