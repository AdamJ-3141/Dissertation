import os
import time
import numpy as np
from .shot_generator import ShotGenerator
from .optimiser import ShotOptimizer
from .evaluator import TableEvaluator
from match import Match, TurnState
import random



class ShotPlanner:
    def __init__(self, sim, evaluator, weights):
        self.sim = sim
        self.evaluator = evaluator
        self.w = self.evaluator.w

        if self.evaluator.colour_set is None:
            # Open Table: Can target any Red (1) or Yellow (2)
            target_set = [i for i in range(1, self.sim.n_obj_balls + 1)
                          if self.sim.colours[i] in (1, 2) and self.sim.in_play[i]]
        else:
            # Groups Decided: Target only our assigned colour
            target_set = [i for i in range(1, self.sim.n_obj_balls + 1)
                          if self.sim.colours[i] == self.evaluator.colour_set
                          and self.sim.in_play[i]]

            # If we have potted all our colours, target the Black (3)
            if not target_set:
                target_set = [i for i in range(1, self.sim.n_obj_balls + 1)
                              if self.sim.colours[i] == 3 and self.sim.in_play[i]]

        self.generator = ShotGenerator(sim, target_set)
        self.optimizer = ShotOptimizer(sim)

    def find_best_shot(self, renderer):
        """
        Refactored decision logic to integrate with ShotOptimizer verification
        and customizable agent weights.
        """
        offensive_result, value = self._find_best_offensive_shot(renderer)

        if offensive_result:
            best_shot_dict, best_params = offensive_result
            if value > self.w.get('aggression_threshold', 5.0):
                return "offensive", best_params

        safety_params = self._find_best_safety_shot()
        return "safety", safety_params

    def _find_best_safety_shot(self, renderer=None):

        candidates = self.generator.get_safety_candidates(num_angles=4)
        if not candidates:
            candidates = self.generator.get_1_cushion_escapes()

        if not candidates:
            return (0.0, 0.5, 0.0, 0.0, 0.0)

        if len(candidates) > 10:
            import random
            random.shuffle(candidates)
            candidates = candidates[:10]

        powers = [0.5, 1.0, 1.5, 2.5]
        spins = [
            (0.0, 0.0), (0.0, 0.5), (0.0, -0.5),
            (0.5, 0.0), (-0.5, 0.0), (0.4, 0.4),
            (0.4, -0.4), (-0.4, 0.4), (-0.4, -0.4)
        ]

        best_safety_score = -float('inf')
        best_params = (0.0, 0.5, 0.0, 0.0, 0.0)

        for shot in candidates:
            aim_angle = shot["seed_angle"]

            for power in powers:
                for topspin, sidespin in spins:

                    vx = power * np.cos(aim_angle)
                    vy = power * np.sin(aim_angle)

                    min_el = None
                    if power > 1e-4:
                        for test_e in range(0, 90, 5):
                            if self.sim.validate_shot(vx, vy, topspin, sidespin, float(test_e)):
                                min_el = float(test_e)
                                break

                    if min_el is None:
                        continue

                    min_el += 3.0
                    params = (aim_angle, power, topspin, sidespin, min_el)

                    # Route the safety shot through the Monte Carlo evaluator!
                    # iterations is usually enough for a fast safety check
                    avg_score = self._get_monte_carlo_score(shot, params, iterations=3, renderer=renderer)

                    if avg_score > best_safety_score:
                        best_safety_score = avg_score
                        best_params = params

        return best_params

    def _find_best_offensive_shot(self, renderer=None):
        """
        Evaluates Direct Pots and Plants using the Numba solvers,
        running a Monte Carlo evaluation on the valid physical executions.
        """

        # Get ONLY Direct and Plant shots from the generator
        candidates = self.generator.get_all_shots()

        candidates.sort(key=lambda x: x.get("efficiency", 0), reverse=True)
        candidates = candidates[:5]

        best_sequence = None
        max_sequence_value = -1e9

        # Early Exit Threshold: If a shot average is massive, take it instantly
        exit_threshold = self.w.get("early_exit_score", 15.0)

        total = len(candidates)

        for idx, shot in enumerate(candidates):

            # Ask Numba solver for the exact aim angle for all Spin/Power combos
            valid_executions = self.optimizer.optimize_shot(shot, renderer)

            if not valid_executions:
                continue

            # Loop through each valid variation (e.g. Center vs Topspin)
            for exec_data in valid_executions:
                params = (
                    exec_data["aim_angle"],
                    exec_data["power"],
                    exec_data["tip_y"],  # topspin
                    exec_data["tip_x"],  # sidespin
                    exec_data["elevation"]
                )

                # Evaluate the shot via Monte Carlo error distribution
                avg_score = self._get_monte_carlo_score(shot, params, iterations=4, renderer=renderer)
                if avg_score > max_sequence_value:
                    max_sequence_value = avg_score
                    best_sequence = (shot, params)
                if avg_score > exit_threshold:
                    # print(f"\n  > Found high-quality shot ({avg_score:.2f}). Exiting search.")
                    return best_sequence, avg_score
                # print(f"MC solver took {time.time() - t2} seconds for shot {params}")
            # print(f"MC solver took {time.time() - t1} seconds.")
        # print(f"Best Shot: {best_sequence}\nAverage Score: {max_sequence_value}")
        return best_sequence, max_sequence_value

    def _get_monte_carlo_score(self, shot, params, iterations=30, renderer=None):
        total_score = 0
        foul_penalty = -2.5
        power_mag = params[1]

        # Base error + (scaling factor * power)
        s_angle = np.radians(0.0 + (0.01 * power_mag))
        s_power = power_mag * 0.02
        s_spin_r = 0.01 + (0.01 * power_mag)
        s_el = 0.2 + (0.02 * power_mag)

        my_color = self.evaluator.colour_set
        opp_color = 2 if my_color == 1 else 1
        cb_endpoints = []

        iter_scores = []

        target_idx = shot["target_idx"]
        target_pocket = shot.get("pocket_idx")

        for i in range(iterations):
            n_angle = np.random.normal(params[0], s_angle)
            n_power = np.random.normal(params[1], s_power)
            n_spin_r = np.random.normal(0, s_spin_r)
            n_spin_angle = np.random.uniform(0, 2 * np.pi)
            n_el = np.random.normal(params[4], s_el)

            self.sim.save_state()

            # PRE-SHOT VISUALIZATION (Draw Intended Path)
            if renderer:
                import pygame
                pygame.event.pump()

                renderer.render(flip=False)

                cb_pos_raw = self.sim.positions[0][:2]
                gb_raw = shot.get("ghost_ball_pos")

                if gb_raw is not None:
                    # Draw pure geometric straight line (Red)
                    pt1 = renderer.world_to_screen(cb_pos_raw)
                    pt2 = renderer.world_to_screen(gb_raw)
                    pygame.draw.line(renderer.screen, (255, 50, 50), pt1, pt2, 3)

                    # Draw Geometric Ghost Ball (Red outline)
                    gb_radius = int(self.sim.cb_radius * renderer.scale)
                    pygame.draw.circle(renderer.screen, (255, 50, 50), (int(pt2[0]), int(pt2[1])), gb_radius, 2)

                if "target_pt" in shot:
                    # Draw pure geometric OB path (Orange)
                    ob_pos_raw = self.sim.positions[target_idx][:2]
                    target_pt_raw = shot["target_pt"]
                    pt1 = renderer.world_to_screen(ob_pos_raw)
                    pt2 = renderer.world_to_screen(target_pt_raw)
                    pygame.draw.line(renderer.screen, (255, 120, 0), pt1, pt2, 3)

                # Calculate Intended Velocities (without noise)
                vx_int = params[1] * np.cos(params[0])
                vy_int = params[1] * np.sin(params[0])

                # Map the Cue Ball Path
                cb_traj = self.sim.map_to_first_coll(vx_int, vy_int, params[2], params[3], params[4])

                if len(cb_traj) > 0:
                    if cb_traj[-1][0] > 100:
                        cb_traj = cb_traj[:-1]

                    if len(cb_traj) > 0:
                        cb_screen_pts = [renderer.world_to_screen(p[:2]) for p in cb_traj]
                        cb_screen_pts.insert(0, renderer.world_to_screen(self.sim.positions[0][:2]))

                        if len(cb_screen_pts) >= 2:
                            pygame.draw.lines(renderer.screen, (255, 255, 255), False, cb_screen_pts, 2)

                        # Draw Physics Impact Point (Light Blue outline)
                        print(f"Optimised ghost ball position: {cb_traj[-1][:2]}")
                        ghost_pos = renderer.world_to_screen(cb_traj[-1][:2])
                        gb_radius = int(self.sim.cb_radius * renderer.scale)
                        pygame.draw.circle(renderer.screen, (200, 200, 255), (int(ghost_pos[0]), int(ghost_pos[1])),
                                           gb_radius, 1)

                from planner.aim_solver import get_solver_trajectory
                import math

                p0_x, p0_y = self.sim.positions[0][:2]
                gb_raw = shot.get("ghost_ball_pos")

                if gb_raw is not None:
                    # Calculate correct initial velocity and spin (matching optimizer math)
                    elevation_rad = math.radians(params[4])
                    v_mag = params[1] * math.cos(elevation_rad)
                    pinch_factor = 1.0 + (math.sin(elevation_rad) ** 2) * 4.5
                    base_spin = (2.5 * params[1] * pinch_factor) / self.sim.cb_radius

                    w_roll = params[2] * base_spin
                    w_z_raw = params[3] * base_spin
                    w_dir = w_z_raw * math.sin(elevation_rad)

                    # Generate the mathematical curve
                    solver_pts = get_solver_trajectory(
                        p0_x, p0_y, gb_raw[0], gb_raw[1],
                        v_mag, params[0], w_roll, w_dir,
                        self.sim.mu_s, self.sim.mu_r, 9.81, self.sim.cb_radius
                    )

                    if len(solver_pts) >= 2:
                        green_screen_pts = [renderer.world_to_screen(p) for p in solver_pts]
                        pygame.draw.lines(renderer.screen, (50, 255, 50), False, green_screen_pts, 2)

                        math_ghost_pos = renderer.world_to_screen(solver_pts[-1])
                        gb_radius = int(self.sim.cb_radius * renderer.scale)
                        pygame.draw.circle(renderer.screen, (50, 255, 50),
                                           (int(math_ghost_pos[0]), int(math_ghost_pos[1])), gb_radius, 2)

                pygame.display.set_caption(f"MC {i + 1}/{iterations} | INTENDED PATH | PRESS SPACE TO SHOOT")
                pygame.display.flip()
                renderer.wait_for_space()

            # SINGLE EXECUTION (With Noise)
            vx = n_power * np.cos(n_angle)
            vy = n_power * np.sin(n_angle)

            self.sim.strike_cue_ball(
                vx, vy,
                topspin_offset=params[2] + n_spin_r * np.sin(n_spin_angle),
                sidespin_offset=params[3] + n_spin_r * np.cos(n_spin_angle),
                elevation_deg=n_el
            )

            if renderer:
                print(
                    f"    MC Attempt [{i + 1}/{iterations}] | Speed: {n_power:.2f} m/s |"
                    f" Angle Offset: {np.degrees(n_angle - params[0]):.2f}° | Spin Radius: {n_spin_r:.2f} |"
                    f" Spin Angle: {n_spin_angle} | Elevation: {n_el}")

                def mc_render_callback(sim_inst):
                    renderer.render(fps=60, flip=True)

                shot_data = self.sim.run(until_first_coll=False, framerate=60, frame_callback=mc_render_callback)
            else:
                shot_data = self.sim.run(until_first_coll=False)

            # SINGLE EVALUATION
            temp_match = Match(self.sim, custom_setup=True)
            original_turn = temp_match.turn
            temp_match.player_colours[original_turn] = my_color
            temp_match.player_colours[1 - original_turn] = opp_color

            temp_match.open_table = (my_color is None)

            temp_match.evaluate_shot(shot_data)

            is_foul = temp_match.turn_state in [TurnState.BALL_IN_HAND, TurnState.BALL_IN_HAND_BAULK]
            is_game_over = temp_match.turn_state == TurnState.GAME_OVER
            is_win = is_game_over and temp_match.winner == original_turn
            is_loss = is_game_over and temp_match.winner != original_turn

            exact_pot = False
            for ev in shot_data["event_history"]:
                if ev[0] == "pot" and ev[1] == target_idx and ev[2] == target_pocket:
                    exact_pot = True
                    break

            if is_loss:
                status = "loss"
                iter_scores.append(-10)
            elif is_win:
                status = "win"
                iter_scores.append(3)
            elif is_foul:
                status = "foul"
                iter_scores.append(foul_penalty)
            elif temp_match.turn == original_turn:
                status = "success" if exact_pot else "fluke"
                temp_eval = TableEvaluator(self.sim, my_color, self.evaluator.w)
                iter_scores.append(temp_eval.get_table_clearability_score())
            else:
                status = "miss"
                opp_eval = TableEvaluator(self.sim, opp_color, self.evaluator.w)
                iter_scores.append(-opp_eval.get_table_clearability_score())

            cb_endpoints.append((self.sim.positions[0][:2].copy(), status))

            # POST-SHOT VISUALIZATION (Cloud)
            if renderer:
                print(f"      -> Result: {status.upper()} | Sub-Scores: {[round(i, 3) for i in iter_scores]}")

                renderer.render(flip=False)
                cloud_surface = pygame.Surface(renderer.screen.get_size(), pygame.SRCALPHA)

                for pos, stat in cb_endpoints:
                    screen_pos = renderer.world_to_screen(pos)
                    if stat == "success":
                        color = (0, 255, 255, 180)
                    elif stat == "foul":
                        color = (255, 0, 0, 180)
                    else:
                        color = (255, 165, 0, 180)
                    pygame.draw.circle(cloud_surface, color, (int(screen_pos[0]), int(screen_pos[1])), 6)

                renderer.screen.blit(cloud_surface, (0, 0))
                pygame.display.set_caption(f"MC {i + 1}/{iterations} | {status.upper()} | PRESS SPACE FOR NEXT")
                pygame.display.flip()

                renderer.wait_for_space()

            # RESTORE STATE
            self.sim.load_state()

        if renderer:
            print(f"  => Final MC Average Score: {total_score / iterations:.2f}\n")

        avg_score = sum(iter_scores) / iterations
        max_score = max(iter_scores)

        # Fetch the risk multiplier (0.0 = strict average, 1.0 = optimism)
        w_risk = self.w.get("w_risk", 0.2)

        # If the best iteration is better than the average, skew the final score upward
        if max_score > avg_score:
            final_score = avg_score + (w_risk * (max_score - avg_score))
        else:
            final_score = avg_score

        if renderer:
            print(f"  => Final MC Average Score: {avg_score:.2f} | Risk-Adjusted: {final_score:.2f}\n")

        return final_score

    def _execute_shot(self, params):
        """Helper to physically execute a shot in the simulation."""
        aim_angle, power, topspin, sidespin, elevation = params
        vx = power * np.cos(aim_angle)
        vy = power * np.sin(aim_angle)

        self.sim.strike_cue_ball(vx, vy, topspin, sidespin, elevation)
        self.sim.run()