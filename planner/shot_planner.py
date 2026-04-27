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

        # --- NEW: Correct Open Table Logic ---
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
        offensive_result = self._find_best_offensive_shot(renderer)

        if offensive_result:
            best_shot_dict, best_params = offensive_result
            current_clearability = self.evaluator.get_table_clearability_score()
            if current_clearability > self.w.get('aggression_threshold', 5.0):
                return "offensive", best_params

        safety_params = self._find_best_safety_shot()
        return "safety", safety_params

    def _find_best_safety_shot(self, renderer=None):

        # 1. Ask the generator for fewer angles per object ball
        candidates = self.generator.get_safety_candidates(num_angles=4)

        if not candidates:
            # We cannot directly see any balls. Try 1-cushion escapes!
            candidates = self.generator.get_1_cushion_escapes()

        if not candidates:
            return (0.0, 0.5, 0.0, 0.0, 0.0)  # Fallback

        # 2. Hard limit the total candidates evaluated to prevent combinatorial explosions
        if len(candidates) > 10:
            random.shuffle(candidates)
            candidates = candidates[:10]

        # 3. Use lower power levels (reduces error natively) and standard spins
        powers = [0.5, 1.0, 1.5, 2.5]
        spins = [
            (0.0, 0.0),  # Center
            (0.0, 0.5),  # Top
            (0.0, -0.5),  # Bottom
            (0.5, 0.0),  # Right
            (-0.5, 0.0),  # Left
            (0.4, 0.4),  # Top Right
            (0.4, -0.4),  # Bottom Right
            (-0.4, 0.4),  # Top Left
            (-0.4, -0.4)  # Bottom Left
        ]

        my_color = self.evaluator.colour_set
        opp_color = 2 if my_color == 1 else 1

        best_safety_score = -float('inf')
        best_safety_params = None

        best_desperation_score = -float('inf')
        best_desperation_params = (0.0, 0.5, 0.0, 0.0, 0.0)

        for shot in candidates:
            # FIX: Use seed_angle directly from the generator! (No target_pt KeyError)
            aim_angle = shot["seed_angle"]

            for power in powers:
                for topspin, sidespin in spins:

                    vx = power * np.cos(aim_angle)
                    vy = power * np.sin(aim_angle)

                    min_el = None
                    if power > 1e-4:
                        for test_e in range(0, 90, 5):
                            # Use the real table to check if the cue hits anything
                            if self.sim.validate_shot(vx, vy, topspin, sidespin, float(test_e)):
                                min_el = float(test_e)
                                break

                    # If the shot is impossible even at 85 degrees, cull it
                    if min_el is None:
                        continue

                    min_el += 3.0  # Add safety margin
                    params = (aim_angle, power, topspin, sidespin, min_el)

                    # --- Single deterministic simulation (NO MONTE CARLO) ---
                    self.sim.save_state()
                    self.sim.strike_cue_ball(vx, vy, topspin, sidespin, min_el)
                    shot_data = self.sim.run(until_first_coll=False)

                    # Evaluate using original logic
                    temp_match = Match(self.sim, custom_setup=True)
                    original_turn = temp_match.turn
                    temp_match.player_colours[original_turn] = my_color
                    temp_match.player_colours[1 - original_turn] = opp_color

                    temp_match.evaluate_shot(shot_data)

                    is_foul = temp_match.turn_state in [TurnState.BALL_IN_HAND, TurnState.BALL_IN_HAND_BAULK]
                    score = 0.0

                    if is_foul:
                        score -= 50.0
                    elif temp_match.turn == original_turn:
                        # We fluked a ball! Evaluate our new setup
                        temp_eval = TableEvaluator(self.sim, my_color, self.w)
                        score += temp_eval.get_table_clearability_score()
                    else:
                        # Evaluate the pain inflicted on the opponent
                        opp_eval = TableEvaluator(self.sim, opp_color, self.w)
                        score -= opp_eval.get_table_clearability_score()

                        visible_balls = 0
                        opp_targets = [i for i in range(1, self.sim.n_obj_balls + 1) if
                                       self.sim.colours[i] in (opp_color, 3) and self.sim.in_play[i]]

                        for op_idx in opp_targets:
                            cb_pos = self.sim.positions[0][:2]
                            op_pos = self.sim.positions[op_idx][:2]

                            if not self.generator.is_cb_path_blocked(cb_pos, op_pos, ignore_indices=[0, op_idx]):
                                visible_balls += 1

                        if visible_balls == 0:
                            score += 100.0  # Devastating total snooker
                        elif visible_balls == 1:
                            score += 30.0  # Great safety, only one target visible
                        else:
                            score -= (visible_balls * 2.0)

                    self.sim.load_state()

                    # Always track the least terrible shot
                    if score > best_desperation_score:
                        best_desperation_score = score
                        best_desperation_params = params

                    if not is_foul and score > best_safety_score:
                        best_safety_score = score
                        best_safety_params = params

        if best_safety_params:
            return best_safety_params

        return best_desperation_params

    def _find_best_offensive_shot(self, renderer=None):
        """
        Evaluates Direct Pots and Plants using the Numba solvers,
        running a Monte Carlo evaluation on the valid physical executions.
        """

        # 1. Get ONLY Direct and Plant shots from the generator
        candidates = self.generator.get_all_shots()

        candidates.sort(key=lambda x: x.get("efficiency", 0), reverse=True)
        candidates = candidates[:5]

        best_sequence = None
        max_sequence_value = -1e9

        # Early Exit Threshold: If a shot average is massive, take it instantly
        exit_threshold = self.w.get("early_exit_score", 15.0)

        total = len(candidates)
        # print(f"\n--- Evaluating {total} Direct/Plant Candidates ---")

        for idx, shot in enumerate(candidates):
            # print(f"  [{idx + 1}/{total}] Ball {shot['target_idx']} to Pocket {shot['pocket_idx']} ({shot['type']})")

            # 2. Ask Numba solver for the exact aim angle for all Spin/Power combos
            valid_executions = self.optimizer.optimize_shot(shot, renderer)

            if not valid_executions:
                # print("      FAILED (No physical path)")
                continue

            # print(f"      FOUND {len(valid_executions)} valid executions. Running Monte Carlo...")
            t1 = time.time()
            # 3. Loop through each valid variation (e.g. Center vs Topspin)
            for exec_data in valid_executions:
                t2 = time.time()
                params = (
                    exec_data["aim_angle"],
                    exec_data["power"],
                    exec_data["tip_y"],  # topspin
                    exec_data["tip_x"],  # sidespin
                    exec_data["elevation"]
                )

                # 4. Evaluate the shot via Monte Carlo error distribution
                avg_score = self._get_monte_carlo_score(shot, params, iterations=4, renderer=renderer)
                if avg_score > max_sequence_value:
                    max_sequence_value = avg_score
                    best_sequence = (shot, params)
                if avg_score > exit_threshold:
                    # print(f"\n  > Found high-quality shot ({avg_score:.2f}). Exiting search.")
                    return best_sequence
                # print(f"MC solver took {time.time() - t2} seconds for shot {params}")
            # print(f"MC solver took {time.time() - t1} seconds.")
        # print(f"Best Shot: {best_sequence}\nAverage Score: {max_sequence_value}")
        return best_sequence

    def _get_monte_carlo_score(self, shot, params, iterations=30, renderer=None):
        total_score = 0
        foul_penalty = -50.0

        # Human Error Standard Deviations
        s_angle = np.radians(0.3)
        s_power = params[1] * 0.03
        s_spin_r = 0.05
        s_el = 1.0

        my_color = self.evaluator.colour_set
        opp_color = 2 if my_color == 1 else 1

        # Track final cue ball positions AND their outcome status
        cb_endpoints = []

        target_idx = shot["target_idx"]
        target_pocket = shot.get("pocket_idx")

        for i in range(iterations):
            n_angle = np.random.normal(params[0], s_angle)
            n_power = np.random.normal(params[1], s_power)
            n_spin_r = np.random.normal(0, s_spin_r)
            n_spin_angle = np.random.uniform(0, 2 * np.pi)
            n_el = np.random.normal(params[4], s_el)

            self.sim.save_state()

            vx = n_power * np.cos(n_angle)
            vy = n_power * np.sin(n_angle)

            self.sim.strike_cue_ball(
                vx, vy,
                topspin_offset=params[2] + n_spin_r * np.sin(n_spin_angle),
                sidespin_offset=params[3] + n_spin_r * np.cos(n_spin_angle),
                elevation_deg=n_el
            )

            shot_data = self.sim.run(until_first_coll=False)
            if any([self.sim.colours[i] == 3 for i in shot_data["balls_potted"]]) and any(
                    [self.sim.in_play[i] and b == my_color for i, b in enumerate(self.sim.colours)]):
                total_score -= 500
            # 1. Evaluate Match Rules
            temp_match = Match(self.sim, custom_setup=True)
            original_turn = temp_match.turn
            temp_match.player_colours[original_turn] = my_color
            temp_match.player_colours[1 - original_turn] = opp_color

            temp_match.evaluate_shot(shot_data)
            is_foul = temp_match.turn_state in [TurnState.BALL_IN_HAND, TurnState.BALL_IN_HAND_BAULK]

            # 2. Check for Exact Intended Pot in the Event History
            exact_pot = False
            for ev in shot_data["event_history"]:
                # Events are structured as ("pot", ball_idx, pocket_idx)
                if ev[0] == "pot" and ev[1] == target_idx and ev[2] == target_pocket:
                    exact_pot = True
                    break

            # 3. Determine Outcome Status
            if is_foul:
                status = "foul"
                total_score += foul_penalty
            elif temp_match.turn == original_turn:
                # We kept the turn, meaning we potted a legal ball
                status = "success" if exact_pot else "fluke"
                temp_eval = TableEvaluator(self.sim, my_color, self.evaluator.w)
                total_score += temp_eval.get_table_clearability_score()
            else:
                # We missed entirely and passed the turn
                status = "miss"
                opp_eval = TableEvaluator(self.sim, opp_color, self.evaluator.w)
                total_score -= opp_eval.get_table_clearability_score()

            # Record position and status
            cb_endpoints.append((self.sim.positions[0][:2].copy(), status))

            self.sim.load_state()

            if renderer and i % 5 == 0:
                import pygame
                pygame.event.pump()

        if renderer:
            import pygame
            renderer.render(flip=False)

            cloud_surface = pygame.Surface(renderer.screen.get_size(), pygame.SRCALPHA)

            for pos, status in cb_endpoints:
                screen_pos = renderer.world_to_screen(pos)

                # Assign colors based on the outcome
                if status == "success":
                    color = (0, 255, 255, 120)  # Cyan for perfect pot
                elif status == "foul":
                    color = (255, 0, 0, 120)  # Red for scratch/foul
                else:
                    color = (255, 165, 0, 120)  # Orange for miss/fluke

                pygame.draw.circle(cloud_surface, color, (int(screen_pos[0]), int(screen_pos[1])), 6)

            renderer.screen.blit(cloud_surface, (0, 0))
            pygame.display.set_caption(f"MC Cloud ({iterations} shots) | Avg Score: {total_score / iterations:.2f}")
            pygame.display.flip()

            pygame.time.wait(1000)
            pygame.event.clear()

        return total_score / iterations

    def _execute_shot(self, params):
        """Helper to physically execute a shot in the simulation."""
        aim_angle, power, topspin, sidespin, elevation = params
        vx = power * np.cos(aim_angle)
        vy = power * np.sin(aim_angle)

        self.sim.strike_cue_ball(vx, vy, topspin, sidespin, elevation)
        self.sim.run()