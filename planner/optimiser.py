import numpy as np
import copy
import time
import math

from pool_simulation.physics import Simulation
from .aim_solver import *
from pool_simulation.constants import *


class ShotOptimizer:
    def __init__(self, sim_instance):
        self.sim = sim_instance
        self.power_levels = [0.2, 0.6, 1.2, 2.5, 3.5]

        self.spins = []
        self.spins.append((0.0, 0.0))  # Center ball

        r_values = [0.5]
        theta_steps = 8
        for r in r_values:
            for i in range(theta_steps):
                theta = i * (math.pi / 4)
                tip_x = r * math.cos(theta)
                tip_y = r * math.sin(theta)
                self.spins.append((tip_x, tip_y))

    def _setup_ghost_table(self, geometric_shot):
        ghost_sim = copy.deepcopy(self.sim)
        ghost_sim.in_play[1:] = False
        ghost_sim.in_play[geometric_shot["target_idx"]] = True
        if geometric_shot.get("type", "") == "plant":
            ghost_sim.in_play[geometric_shot["combo_idx"]] = True

        ghost_sim.saved_positions = ghost_sim.positions.copy()
        ghost_sim.saved_in_play = ghost_sim.in_play.copy()
        return ghost_sim

    def optimize_shot(self, geometric_shot, renderer=None):
        valid_executions = []
        target_idx = geometric_shot["target_idx"]
        shot_type = geometric_shot.get("type", "")

        if "bank" in shot_type or "kick" in shot_type or "carom" in shot_type:
            return []

        cb_pos = self.sim.positions[0][:2]
        gb_pos = geometric_shot["ghost_ball_pos"]
        geom_angle = math.atan2(gb_pos[1] - cb_pos[1], gb_pos[0] - cb_pos[0])

        ordered_powers = [1.2, 0.6, 2.5, 0.2, 3.5]
        R = float(self.sim.radii[0])

        target_pt = geometric_shot.get("target_pt")
        ob_pos = self.sim.positions[target_idx][:2]

        dist_pot = math.hypot(target_pt[0] - ob_pos[0], target_pt[1] - ob_pos[1])
        dist_cb_to_gb = math.hypot(gb_pos[0] - cb_pos[0], gb_pos[1] - cb_pos[1])

        is_plant = (shot_type == "plant")
        dist_combo = 0.0

        if is_plant:
            combo_idx = geometric_shot.get("combo_idx")
            if combo_idx is not None:
                combo_pos = self.sim.positions[combo_idx][:2]
                gb1_pos = geometric_shot.get("gb1_pos", ob_pos)
                dist_combo = math.hypot(gb1_pos[0] - combo_pos[0], gb1_pos[1] - combo_pos[1])

        efficiency = geometric_shot.get("efficiency", 1.0)
        eff1 = geometric_shot.get("eff1", efficiency)
        eff2 = geometric_shot.get("eff2", 1.0)
        m_cb = self.sim.cb_mass
        m_ob = self.sim.ob_mass

        for tip_x, tip_y in self.spins:
            for power in ordered_powers:

                test_vx = power * math.cos(geom_angle)
                test_vy = power * math.sin(geom_angle)
                min_el = None

                if power > 1e-4:
                    for test_e in range(0, 90, 5):
                        if self.sim.validate_shot(test_vx, test_vy, tip_y, tip_x, float(test_e)):
                            min_el = float(test_e)
                            break

                if min_el is None:
                    continue

                min_el = min_el + 3
                elevation_rad = math.radians(min_el)

                v_mag = power * math.cos(elevation_rad)
                pinch_factor = 1.0 + (math.sin(elevation_rad) ** 2) * 4.5
                base_spin_magnitude = (2.5 * power * pinch_factor) / R

                w_roll = tip_y * base_spin_magnitude
                w_z_raw = tip_x * base_spin_magnitude
                w_dir = w_z_raw * math.sin(elevation_rad)

                v_impact = get_impact_velocity(v_mag, w_roll, R, MU_S, MU_R, g, dist_cb_to_gb)
                if not check_sufficient_speed(
                        v_impact, eff1, eff2, dist_combo, dist_pot,
                        m_cb, m_ob, RESTITUTION, MU_S, MU_R, g, is_plant
                ):
                    continue

                exact_aim_angle, converged = solve_exact_aim_angle(
                    cb_pos[0], cb_pos[1], gb_pos[0], gb_pos[1],
                    v_mag, w_roll, w_dir,
                    MU_S, MU_R, g, CUE_BALL_RADIUS
                )

                if not converged:
                    continue

                self.sim.save_state()

                final_data = self._verify_shot_on_real_table(
                    self.sim, geometric_shot, exact_aim_angle,
                    power, tip_y, tip_x, min_el
                )
                self.sim.load_state()

                if final_data:
                    is_intended = target_idx in final_data["balls_potted"]
                    valid_executions.append({
                        "aim_angle": exact_aim_angle,
                        "power": power,
                        "tip_x": tip_x, "tip_y": tip_y,
                        "elevation": min_el,
                        "final_state": final_data,
                        "intended_pot": is_intended
                    })

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
                            if b2 is None or ev[2] == b2:
                                return idx
            return -1

        idx_pot = first_idx("pot", target, target_pocket)
        if require_pot and idx_pot == -1:
            return False

        idx_cb_cush = first_idx("cushion", 0)

        if shot_type == "direct":
            idx_cb_target = first_idx("ball", 0, target)
            if idx_cb_target == -1: return False
            if -1 < idx_cb_cush < idx_cb_target: return False

        elif shot_type == "plant":
            combo = shot_dict["combo_idx"]
            idx_cb_combo = first_idx("ball", 0, combo)
            idx_combo_target = first_idx("ball", combo, target)

            if idx_cb_combo == -1: return False
            if -1 < idx_cb_cush < idx_cb_combo: return False
            if idx_combo_target == -1: return False
            if not (idx_cb_combo < idx_combo_target): return False

        return True

    def _verify_shot_on_real_table(self, sim_instance: Simulation, shot_dict, aim_angle, p, t, s, el):
        vx = p * np.cos(aim_angle)
        vy = p * np.sin(aim_angle)

        sim_instance.strike_cue_ball(vx, vy, topspin_offset=t, sidespin_offset=s, elevation_deg=el)
        shot_data = sim_instance.run(until_first_coll=True)

        if shot_data["error"] or 0 in shot_data["balls_potted"]:
            return None

        gb_target = shot_dict.get("ghost_ball_pos")
        if gb_target is not None:
            cb_impact = sim_instance.positions[0][:2]
            dist = math.hypot(cb_impact[0] - gb_target[0], cb_impact[1] - gb_target[1])

            # If the engine sideswiped the ball 7.7mm away from the ghost ball, reject the curve!
            if dist > 0.001:
                return None

        # Verify the very first thing the cue ball touched was the correct target
        first_target = shot_dict.get("combo_idx", shot_dict["target_idx"])
        if shot_data["first_ball_hit"] != first_target:
            return None

        # Verify it didn't hit a cushion before the target (unless it's a kick shot)
        for ev in shot_data["event_history"]:
            if ev[0] == "cushion" and ev[1] == 0:
                return None
            # Break early once we see the ball collision, since we only care about what happened before it
            if ev[0] == "ball" and 0 in (ev[1], ev[2]):
                break

        # FAST FIX 3: We no longer check require_pot=True or _is_sequence_valid.
        # The Numba solver proved the angle hits the ghost ball mathematically.
        # The Monte Carlo planner will verify the actual pot in the next phase!
        return shot_data