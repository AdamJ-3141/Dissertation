import time

import numpy as np
import copy
import math


def _distance_point_to_line_segment(p, a, b):
    """Calculates the minimum distance from point p to the line segment ab."""
    ab = b - a
    ap = p - a

    norm_ab_sq = np.dot(ab, ab)
    if norm_ab_sq < 1e-8:
        return np.linalg.norm(p - a)

    t = max(0.0, min(1.0, np.dot(ap, ab) / norm_ab_sq))
    closest_point = a + t * ab

    return np.linalg.norm(p - closest_point)

class ShotOptimizer:
    def __init__(self, sim_instance):
        self.sim = sim_instance

        # 1. Power Levels
        self.power_levels = [0.2, 0.6, 1.2, 2.5, 3.5]

        # 2. Spin Grid (Polar to Cartesian)
        self.spins = []
        self.spins.append((0.0, 0.0))  # R = 0

        r_values = [0.5]
        theta_steps = 8
        for r in r_values:
            for i in range(theta_steps):
                theta = i * (math.pi / 4)  # 0 to 11pi/6
                tip_x = r * math.cos(theta)  # sidespin
                tip_y = r * math.sin(theta)  # topspin
                self.spins.append((tip_x, tip_y))

        self.elevation_offsets = [0.0, 5.0, 10.0]

    def _setup_ghost_table(self, geometric_shot):
        """Creates an isolated simulation containing ONLY the required balls."""
        ghost_sim = copy.deepcopy(self.sim)
        ghost_sim.in_play[1:] = False

        # Always keep the Target Ball in play
        ghost_sim.in_play[geometric_shot["target_idx"]] = True

        # Keep intermediate balls based on shot type
        shot_type = geometric_shot["type"]
        if shot_type == "plant":
            ghost_sim.in_play[geometric_shot["combo_idx"]] = True
        elif shot_type == "cue_carom":
            ghost_sim.in_play[geometric_shot["ob1_idx"]] = True
        elif shot_type == "carom":
            ghost_sim.in_play[geometric_shot["kiss_idx"]] = True

        ghost_sim.saved_positions = ghost_sim.positions.copy()
        ghost_sim.saved_in_play = ghost_sim.in_play.copy()

        return ghost_sim

    def _get_cushion_name(self, ev):
        """Translates engine segment indices into human-readable table sides."""
        _, _, target_type, target_idx = ev

        if target_type == 'line':
            p1, p2 = self.sim.line_segments[target_idx]
            cx, cy = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
        elif target_type == 'circle':
            cx, cy = self.sim.circles[target_idx][:2]
        else:
            return "unknown"

        # A diagonal line from the center to a corner pocket has a slope of 0.5 (Width 2, Height 1)
        # By comparing the absolute coordinates to this 0.5 ratio, we instantly know which quadrant we are in!
        if abs(cy) > 0.5 * abs(cx):
            return 'top' if cy > 0 else 'bottom'
        else:
            return 'right' if cx > 0 else 'left'

    def optimize_shot(self, geometric_shot):
        start = time.time()
        valid_executions = []
        target_idx = geometric_shot["target_idx"]
        pocket_target_pt = geometric_shot["target_pt"]
        seed_angle = geometric_shot["seed_angle"]

        base_ghost_sim = self._setup_ghost_table(geometric_shot)

        # 1. Test a "baseline" medium power first.
        # If the path is blocked at 1.2 power, it will be blocked at 2.5 and 0.2.
        ordered_powers = [1.2, 0.6, 2.5, 0.2, 3.5]

        # 2. Outer Loop: Spins (Because spin dictates the physical shape/swerve of the shot)
        for tip_x, tip_y in self.spins:
            spin_path_blocked = False

            # 3. Inner Loop: Powers (Cull these if the baseline fails)
            for power in ordered_powers:
                if spin_path_blocked:
                    break  # Cull remaining powers, move to the next spin

                print(f"      - Testing Spin ({tip_x:.1f}, {tip_y:.1f}) @ Pwr {power:.1f}   ")
                print(f"Elapsed time: {time.time() - start}")
                test_vx = power * np.cos(seed_angle)
                test_vy = power * np.sin(seed_angle)

                min_el = 89.0
                if power > 1e-4:
                    for test_e in range(0, 90, 5):
                        if self.sim.validate_shot(test_vx, test_vy, tip_y, tip_x, float(test_e)):
                            min_el = float(test_e)
                            break

                for el_offset in self.elevation_offsets:
                    actual_elevation = min_el + el_offset
                    if actual_elevation > 89.0: continue

                    print("Finding initial hits")
                    print(f"Elapsed time: {time.time() - start}")
                    guesses = self._find_initial_hits(
                        base_ghost_sim, geometric_shot, pocket_target_pt, seed_angle,
                        power, tip_y, tip_x, actual_elevation
                    )

                    if not guesses:
                        # THE POWER CULL:
                        # If we can't find a path to the target ball at this spin,
                        # changing the power won't fix it. Cull all other powers for this spin.
                        spin_path_blocked = True
                        break  # Break elevation loop
                    print("Secant search starting")
                    print(f"Elapsed time: {time.time() - start}")
                    true_aim_angle = self._secant_search(
                        base_ghost_sim, geometric_shot, pocket_target_pt, guesses,
                        power, tip_y, tip_x, actual_elevation
                    )
                    print("Secant search finished")
                    print(f"Elapsed time: {time.time() - start}")
                    if true_aim_angle is not None:
                        real_sim = copy.deepcopy(self.sim)
                        print("Verifying on real table")
                        print(f"Elapsed time: {time.time() - start}")
                        final_data = self._verify_shot_on_real_table(
                            real_sim, geometric_shot, true_aim_angle,
                            power, tip_y, tip_x, actual_elevation
                        )
                        print("Finished verifying")
                        print(f"Elapsed time: {time.time() - start}")
                        if final_data:
                            is_intended = target_idx in final_data["balls_potted"]
                            valid_executions.append({
                                "aim_angle": true_aim_angle,
                                "power": power,
                                "tip_x": tip_x, "tip_y": tip_y,
                                "elevation": actual_elevation,
                                "final_state": final_data,
                                "intended_pot": is_intended
                            })
                            # Break elevation loop once we find a valid hit for this power/spin
                            break

        print(" " * 60, end="\r", flush=True)
        return valid_executions

    def _is_sequence_valid(self, shot_dict, event_history, require_pot=False):
        shot_type = shot_dict.get("type", "")
        target = shot_dict["target_idx"]
        target_pocket = shot_dict.get("pocket_idx")

        def first_idx(kind, b1, b2=None):
            for idx, ev in enumerate(event_history):
                if ev[0] == kind:
                    if kind == "ball":
                        if (ev[1] == b1 and ev[2] == b2) or (ev[1] == b2 and ev[2] == b1):
                            return idx
                    elif kind == "cushion":
                        if ev[1] == b1:
                            return idx
                    elif kind == "pot":
                        if ev[1] == b1:
                            # Verify it went into the correct pocket!
                            if b2 is None or ev[2] == b2:
                                return idx
            return -1

        idx_pot = first_idx("pot", target, target_pocket)
        if require_pot and idx_pot == -1:
            return False

        # Only track the cue ball's initial cushion hits to prevent Flaw 1
        idx_cb_cush = first_idx("cushion", 0)

        if shot_type == "direct" or "bank" in shot_type:
            idx_cb_target = first_idx("ball", 0, target)
            if idx_cb_target == -1: return False
            if idx_cb_cush > -1 and idx_cb_cush < idx_cb_target: return False

        elif "kick" in shot_type:
            idx_cb_target = first_idx("ball", 0, target)
            if idx_cb_target > -1 and idx_cb_cush > -1:
                if not (idx_cb_cush < idx_cb_target): return False

        elif shot_type == "plant":
            combo = shot_dict["combo_idx"]
            idx_cb_combo = first_idx("ball", 0, combo)
            idx_combo_target = first_idx("ball", combo, target)
            if idx_cb_combo == -1: return False
            if idx_cb_cush > -1 and idx_cb_cush < idx_cb_combo: return False
            # STRICT CULL: The combo ball MUST hit the target ball
            if idx_combo_target == -1: return False
            if not (idx_cb_combo < idx_combo_target): return False

        elif shot_type == "carom":
            kiss = shot_dict["kiss_idx"]
            idx_cb_target = first_idx("ball", 0, target)
            idx_target_kiss = first_idx("ball", target, kiss)
            if idx_cb_target == -1: return False
            if idx_cb_cush > -1 and idx_cb_cush < idx_cb_target: return False
            # STRICT CULL: The target ball MUST hit the kiss ball
            if idx_target_kiss == -1: return False
            if not (idx_cb_target < idx_target_kiss): return False

        elif shot_type == "cue_carom":
            ob1 = shot_dict["ob1_idx"]
            idx_cb_ob1 = first_idx("ball", 0, ob1)
            idx_cb_target = first_idx("ball", 0, target)
            if idx_cb_ob1 == -1: return False
            if idx_cb_cush > -1 and idx_cb_cush < idx_cb_ob1: return False
            # STRICT CULL: The cue ball MUST eventually hit the target ball
            if idx_cb_target == -1: return False
            if not (idx_cb_ob1 < idx_cb_target): return False

        return True

    def _get_continuous_error(self, test_sim, shot_dict, target_pt, aim_angle, p, t, s, el, debug=False):
        test_sim.reset(test_sim.saved_positions.copy(), test_sim.colours, test_sim.saved_in_play.copy())

        target_idx = shot_dict["target_idx"]
        first_target = shot_dict.get("ob1_idx", shot_dict.get("combo_idx", target_idx))
        shot_type = shot_dict.get("type", "")

        required_bounces = len(shot_dict.get("bounce_points", [])) if "bank" in shot_type else 0

        actor_idx = None
        receiver_idx = None
        if shot_type == "plant":
            actor_idx = shot_dict["combo_idx"]
            receiver_idx = target_idx
        elif shot_type == "carom":
            actor_idx = target_idx
            receiver_idx = shot_dict["kiss_idx"]
        elif shot_type == "cue_carom":
            actor_idx = 0
            receiver_idx = target_idx

        # ====================================================================
        # 1. RUN ANALYTICALLY (No Callback, No Framerate Bottleneck!)
        # ====================================================================
        vx = p * np.cos(aim_angle)
        vy = p * np.sin(aim_angle)
        test_sim.strike_cue_ball(vx, vy, topspin_offset=t, sidespin_offset=s, elevation_deg=el)

        # Runs instantly to completion
        test_sim.run(until_first_coll=False)

        # ====================================================================
        # 2. EXTRACT THE TRACE
        # ====================================================================
        tracker = {
            "cb_min_dist": float('inf'), "cb_cross": 0.0,
            "sec_min_dist": float('inf'), "sec_cross": 0.0,
            "ob_cross": 1.0, "initial_ob_cross": 0.0,
        }

        trace = test_sim.shot_data.get("trace", [])
        events = test_sim.shot_data.get("event_history", [])

        first_target_pos = test_sim.saved_positions[first_target][:2]
        receiver_pos = test_sim.saved_positions[receiver_idx][:2] if receiver_idx is not None else None

        if trace:
            for i in range(len(trace) - 1):
                pos_start, vel_start, _, in_play_start = trace[i]
                pos_end, _, _, _ = trace[i + 1]

                # 1. Cue Ball to First Target
                cb_start = pos_start[0][:2]
                cb_end = pos_end[0][:2]
                dist_cb = _distance_point_to_line_segment(first_target_pos, cb_start, cb_end)

                if dist_cb < tracker["cb_min_dist"]:
                    tracker["cb_min_dist"] = dist_cb
                    v_cb = vel_start[0][:2]
                    v_cb_norm = np.linalg.norm(v_cb)
                    if v_cb_norm > 1e-6:
                        dp_cb = first_target_pos - cb_start
                        tracker["cb_cross"] = (v_cb[0] * dp_cb[1] - v_cb[1] * dp_cb[0]) / v_cb_norm

                # 2. Actor to Receiver (for plants/caroms)
                if actor_idx is not None and receiver_pos is not None:
                    if in_play_start[actor_idx]:
                        act_start = pos_start[actor_idx][:2]
                        act_end = pos_end[actor_idx][:2]
                        dist_sec = _distance_point_to_line_segment(receiver_pos, act_start, act_end)

                        if dist_sec < tracker["sec_min_dist"]:
                            tracker["sec_min_dist"] = dist_sec
                            v_act = vel_start[actor_idx][:2]
                            v_act_norm = np.linalg.norm(v_act)
                            if v_act_norm > 1e-6:
                                dp_sec = receiver_pos - act_start
                                tracker["sec_cross"] = (v_act[0] * dp_sec[1] - v_act[1] * dp_sec[0]) / v_act_norm

            ob_cushion_hits = sum(1 for ev in events if ev[0] == "cushion" and ev[1] == target_idx)
            ready_for_final_leg = (ob_cushion_hits == required_bounces)

            if shot_type == "carom":
                has_kissed = any(ev[0] == "ball" and
                                 ((ev[1] == target_idx and ev[2] == receiver_idx) or
                                  (ev[2] == target_idx and ev[1] == receiver_idx))
                                 for ev in events)
                ready_for_final_leg = ready_for_final_leg and has_kissed

            # Find initial trajectory
            found_initial = False
            for state in trace:
                pos_state, vel_state, _, _ = state
                v_ob = vel_state[target_idx][:2]
                v_ob_norm = np.linalg.norm(v_ob)

                if v_ob_norm > 1e-4:
                    if not found_initial:
                        pos = pos_state[target_idx][:2]
                        fb_vec = np.array(target_pt[:2]) - pos
                        fb_norm = np.linalg.norm(fb_vec)
                        if fb_norm > 1e-6:
                            fb_dir = fb_vec / fb_norm
                            v_dir = v_ob / v_ob_norm
                            tracker["initial_ob_cross"] = (v_dir[0] * fb_dir[1]) - (v_dir[1] * fb_dir[0])
                        found_initial = True

            # Get final leg trajectory by scanning backwards from the end
            if ready_for_final_leg:
                for state in reversed(trace):
                    pos_state, vel_state, _, _ = state
                    v_ob = vel_state[target_idx][:2]
                    v_ob_norm = np.linalg.norm(v_ob)
                    if v_ob_norm > 1e-4:
                        pos = pos_state[target_idx][:2]
                        ideal_vec = np.array(target_pt[:2]) - pos
                        ideal_norm = np.linalg.norm(ideal_vec)
                        ideal_dir = ideal_vec / ideal_norm if ideal_norm > 1e-6 else np.array([1.0, 0.0])
                        v_dir = v_ob / v_ob_norm
                        tracker["ob_cross"] = (v_dir[0] * ideal_dir[1]) - (v_dir[1] * ideal_dir[0])
                        break

        # ====================================================================
        # 3. REJECTION LOGIC (Untouched from your original code)
        # ====================================================================
        if test_sim.shot_data["first_ball_hit"] != first_target:
            sign = 1.0 if tracker["cb_cross"] > 0 else -1.0
            err = sign * (tracker["cb_min_dist"] + 30.0)
            if debug: print(f"        -> REJECTED (Scenario A): Missed first ball. Err: {err:.2f}")
            return err

        if not self._is_sequence_valid(shot_dict, test_sim.shot_data["event_history"], require_pot=False):
            first_hit_legal = False
            if test_sim.shot_data["first_ball_hit"] == first_target:
                idx_cb_cush = next((i for i, ev in enumerate(test_sim.shot_data["event_history"]) if
                                    ev[0] == "cushion" and ev[1] == 0), -1)
                idx_cb_ball = next(
                    (i for i, ev in enumerate(test_sim.shot_data["event_history"]) if ev[0] == "ball" and ev[1] == 0),
                    -1)

                if "kick" in shot_type:
                    if idx_cb_cush > -1 and (idx_cb_ball == -1 or idx_cb_cush < idx_cb_ball):
                        first_hit_legal = True
                else:
                    if idx_cb_cush == -1 or (-1 < idx_cb_ball < idx_cb_cush):
                        first_hit_legal = True

            if first_hit_legal and actor_idx is not None:
                sign = 1.0 if tracker["sec_cross"] > 0 else -1.0
                err = sign * (tracker["sec_min_dist"] + 20.0)
                if debug: print(f"        -> REJECTED (Scenario B2): Missed 2nd ball. Err: {err:.2f}")
                return err
            else:
                sign = 1.0 if tracker["cb_cross"] > 0 else -1.0
                err = sign * (abs(tracker["cb_cross"]) + 30.0)
                if debug: print(f"        -> REJECTED (Scenario B1): Invalid CB sequence. Err: {err:.2f}")
                return err

        ob_cushion_hits = sum(
            1 for ev in test_sim.shot_data["event_history"] if ev[0] == "cushion" and ev[1] == target_idx)
        if ob_cushion_hits < required_bounces:
            sign = 1.0 if tracker["initial_ob_cross"] > 0 else -1.0
            err = sign * (abs(tracker["initial_ob_cross"]) + 10.0)
            if debug: print(f"        -> REJECTED (Scenario C): Missed cushion. Err: {err:.2f}")
            return err

        if "bank" in shot_type:
            intended_cushion = shot_type.split('_')[-1]
            actual_cushions = [self._get_cushion_name(ev) for ev in test_sim.shot_data["event_history"] if
                               ev[0] == "cushion" and ev[1] == target_idx]

            if len(actual_cushions) > 0 and actual_cushions[0] != intended_cushion:
                sign = 1.0 if tracker["initial_ob_cross"] > 0 else -1.0
                err = sign * (abs(tracker["initial_ob_cross"]) + 10.0)
                if debug: print(f"        -> REJECTED (Scenario D): Wrong Bank Cushion. Err: {err:.2f}")
                return err

        elif "kick" in shot_type:
            intended_sequence = shot_dict.get("sequence", [])
            actual_cushions = [self._get_cushion_name(ev) for ev in test_sim.shot_data["event_history"] if
                               ev[0] == "cushion" and ev[1] == 0]

            for i, expected_cush in enumerate(intended_sequence):
                if i < len(actual_cushions) and actual_cushions[i] != expected_cush:
                    sign = 1.0 if tracker["cb_cross"] > 0 else -1.0
                    err = sign * (abs(tracker["cb_cross"]) + 10.0)
                    if debug: print(f"        -> REJECTED (Scenario D): Wrong Kick Sequence. Err: {err:.2f}")
                    return err

        if debug: print(f"        -> ACCEPTED (Valid Sequence). OB Cross Err: {tracker['ob_cross']:.5f}")
        return tracker["ob_cross"]

    def _find_initial_hits(self, ghost_sim, shot_dict, target_pt, seed_angle, p, t, s, el, debug=False):
        """
        Spirals outward from the seed angle to find a valid contact point,
        then refines to establish a gradient.
        """
        # Step 1: Check the seed immediately
        e_seed = self._get_continuous_error(ghost_sim, shot_dict, target_pt, seed_angle, p, t, s, el)
        if abs(e_seed) < 5.0:
            a1 = seed_angle + 0.001
            e1 = self._get_continuous_error(ghost_sim, shot_dict, target_pt, a1, p, t, s, el)
            return [(seed_angle, e_seed), (a1, e1)]

        # Step 2: Spiral outward in 0.5-degree steps up to 45 degrees
        max_deviation = np.radians(45)
        step_size = np.radians(0.5)
        current_step = 1

        while (current_step * step_size) <= max_deviation:
            # Alternate: +0.5, -0.5, +1.0, -1.0...
            for side in [1, -1]:
                angle = seed_angle + (side * current_step * step_size)
                err = self._get_continuous_error(ghost_sim, shot_dict, target_pt, angle, p, t, s, el)

                # If we connect (error < 5.0 indicates valid sequence hit)
                if abs(err) < 5.0:
                    # Establish a finer gradient near this hit
                    a_fine = angle + 0.001
                    e_fine = self._get_continuous_error(ghost_sim, shot_dict, target_pt, a_fine, p, t, s, el)
                    return [(angle, err), (a_fine, e_fine)]

            current_step += 1

        return None  # No hit found within 45 degrees

    def _secant_search(self, ghost_sim, shot_dict, target_pt, guesses, p, t, s, el, tol=1e-4, max_iter=20, debug=False):
        a0, e0 = guesses[0]
        a1, e1 = guesses[1]

        target_idx = shot_dict["target_idx"]
        max_step = math.radians(5)

        best_pot_angle = None
        best_pot_err = float('inf')

        for _ in range(max_iter):
            if abs(e1 - e0) < 1e-9:
                break

            # Calculate raw secant step
            raw_step = e1 * ((a1 - a0) / (e1 - e0))

            # Clamp the step
            clamped_step = max(min(raw_step, max_step), -max_step)

            a_next = a1 - clamped_step
            e_next = self._get_continuous_error(ghost_sim, shot_dict, target_pt, a_next, p, t, s, el, debug=debug)

            # Check if this exact physics evaluation actually potted the ball
            is_potted = target_idx in ghost_sim.shot_data["balls_potted"]

            # ONLY save it as a valid fallback if it didn't trigger a massive penalty (error <= 1.0)
            if is_potted and abs(e_next) <= 1.0 and abs(e_next) < best_pot_err:
                best_pot_err = abs(e_next)
                best_pot_angle = a_next

            if abs(e_next) < tol:
                if is_potted:
                    return a_next
                else:
                    # Mathematical perfection, but physical failure (e.g., hit the jaw).
                    # Fall back to the closest angle we found that actually dropped.
                    if debug: print(
                        f"        -> REJECTED SECANT ROOT: Physics Miss. Falling back to {math.degrees(best_pot_angle) if best_pot_angle else 'None'}°")
                    return best_pot_angle

            a0, e0 = a1, e1
            a1, e1 = a_next, e_next

        # If we hit max_iter without reaching 0.0 tolerance, but we DID find a pot, use it!
        return best_pot_angle

    def _verify_shot_on_real_table(self, real_sim, shot_dict, aim_angle, p, t, s, el):
        """Runs the final verified shot on the cluttered table to check for flukes or blocked paths."""
        real_sim.reset(real_sim.positions, real_sim.colours, real_sim.in_play)
        vx = p * np.cos(aim_angle)
        vy = p * np.sin(aim_angle)

        real_sim.strike_cue_ball(vx, vy, topspin_offset=t, sidespin_offset=s, elevation_deg=el)
        shot_data = real_sim.run(until_first_coll=False)

        # 1. Did we scratch?
        if 0 in shot_data["balls_potted"] or shot_data["error"]:
            return None

        # 2. Did we hit the right first ball?
        first_target = shot_dict.get("ob1_idx", shot_dict.get("combo_idx", shot_dict["target_idx"]))
        if shot_data["first_ball_hit"] != first_target:
            return None

            # 3. Did the exact intended chronological sequence execute perfectly into the pocket?
        if not self._is_sequence_valid(shot_dict, shot_data["event_history"], require_pot=True):
            return None

        return shot_data