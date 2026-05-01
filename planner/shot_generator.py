import numpy as np
from pool_simulation.constants import *
from pool_simulation.physics import Simulation
from .aim_solver import check_path_obstruction


def _mirror_point(pt, cush, w, h):
    x, y = pt
    if cush == 'left': return -2 * w - x, y
    if cush == 'right': return 2 * w - x, y
    if cush == 'top': return x, 2 * h - y
    if cush == 'bottom': return x, -2 * h - y
    return None


def _get_intersection(p1, p2, cush, w, h):
    x1, y1 = p1
    x2, y2 = p2

    if cush == 'left':
        if x2 == x1: return None
        t = (-w - x1) / (x2 - x1)
        return (-w, y1 + t * (y2 - y1)), t
    elif cush == 'right':
        if x2 == x1: return None
        t = (w - x1) / (x2 - x1)
        return (w, y1 + t * (y2 - y1)), t
    elif cush == 'top':
        if y2 == y1: return None
        t = (h - y1) / (y2 - y1)
        return (x1 + t * (x2 - x1), h), t
    elif cush == 'bottom':
        if y2 == y1: return None
        t = (-h - y1) / (y2 - y1)
        return (x1 + t * (x2 - x1), -h), t
    return None


def _in_bounds(pt, cush, w, h):
    eps = 0.001
    x, y = pt
    if cush in ('left', 'right'):
        return -h - eps <= y <= h + eps
    else:
        return -w - eps <= x <= w + eps


class ShotGenerator:
    def __init__(self, main_sim: Simulation, target_set: list[int]):
        self.main_sim = main_sim
        self.target_set = target_set
        self.ghost_distance = OBJECT_BALL_RADIUS + CUE_BALL_RADIUS

    def get_pocket_targets(self, points_per_pocket=3, spread_ratio=0.6):
        pockets = self.main_sim.pockets
        target_points = []

        for px, py, r in pockets:
            vx = 0.0 if abs(px) < 0.01 else -np.sign(px)
            vy = 0.0 if abs(py) < 0.01 else -np.sign(py)

            length = np.hypot(vx, vy)
            vx /= length
            vy /= length

            mouth_x = px + (vx * r)
            mouth_y = py + (vy * r)

            tx = -vy
            ty = vx

            spread_dist = r * spread_ratio
            offsets = np.linspace(-spread_dist, spread_dist, points_per_pocket)

            pocket_pts = []
            for offset in offsets:
                pocket_pts.append((mouth_x + (tx * offset), mouth_y + (ty * offset)))

            target_points.append(pocket_pts)

        return target_points

    def get_direct_pots(self):
        positions = self.main_sim.positions
        cue_ball_pos = positions[0]
        pocket_targets_list = self.get_pocket_targets()

        direct_shots = []

        for target_idx in self.target_set:
            target_pos = positions[target_idx]

            for pocket_idx, pocket_pts in enumerate(pocket_targets_list):
                for pt_idx, target_pt in enumerate(pocket_pts):

                    # Check OB to Pocket Path (Strict width)
                    if self._is_path_blocked(target_pos, target_pt, OBJECT_BALL_RADIUS, ignore_indices=[target_idx]):
                        continue

                    dx_pot = target_pt[0] - target_pos[0]
                    dy_pot = target_pt[1] - target_pos[1]
                    dist_pot = np.hypot(dx_pot, dy_pot)
                    if dist_pot == 0: continue

                    dir_pot_x = dx_pot / dist_pot
                    dir_pot_y = dy_pot / dist_pot

                    gb_x = target_pos[0] - self.ghost_distance * dir_pot_x
                    gb_y = target_pos[1] - self.ghost_distance * dir_pot_y
                    gb_pos = (gb_x, gb_y)

                    dx_aim = gb_x - cue_ball_pos[0]
                    dy_aim = gb_y - cue_ball_pos[1]
                    dist_aim = np.hypot(dx_aim, dy_aim)
                    if dist_aim == 0: continue

                    # Check CB to GB Path (Slightly reduced width for swerve)
                    if self._is_path_blocked(cue_ball_pos, gb_pos, CUE_BALL_RADIUS * 0.85,
                                             ignore_indices=[0, target_idx]):
                        continue

                    dir_aim_x = dx_aim / dist_aim
                    dir_aim_y = dy_aim / dist_aim

                    efficiency = (dir_aim_x * dir_pot_x) + (dir_aim_y * dir_pot_y)

                    # DYNAMIC DIFFICULTY CULLING (Sliding Scale)
                    total_dist = dist_aim + dist_pot
                    dynamic_max_angle = 85.0 - (total_dist * 15.0)

                    # Clamp it so the AI will always at least consider a 60-degree cut
                    dynamic_max_angle = max(60.0, dynamic_max_angle)

                    # Convert the dynamic angle limit to an efficiency dot-product
                    dynamic_min_efficiency = np.cos(np.radians(dynamic_max_angle))

                    if efficiency <= dynamic_min_efficiency:
                        continue

                    aim_angle = np.arctan2(dy_aim, dx_aim)

                    direct_shots.append({
                        "target_idx": target_idx,
                        "pocket_idx": pocket_idx,
                        "target_pt_idx": pt_idx,
                        "ghost_ball_pos": gb_pos,
                        "seed_angle": aim_angle,
                        "target_pt": target_pt,
                        "type": "direct",
                        "efficiency": efficiency
                    })

        return direct_shots

    def _is_ob_path_blocked(self, start_pos, end_pos, ignore_indices):
        """
        Checks if the straight-line path of the object ball to the pocket
        is blocked by any other ball on the table or by pocket knuckles.
        """
        start_x, start_y = start_pos[0], start_pos[1]
        end_x, end_y = end_pos[0], end_pos[1]

        # Check for ball obstructions
        for i in range(1, self.main_sim.n_obj_balls + 1):
            if i in ignore_indices or not self.main_sim.in_play[i]:
                continue

            b_pos = self.main_sim.positions[i]

            # Both the moving path and the obstacle have the radius of a ball
            if check_path_obstruction(
                    start_x, start_y, end_x, end_y,
                    b_pos[0], b_pos[1],
                    OBJECT_BALL_RADIUS, OBJECT_BALL_RADIUS
            ):
                return True

        # Check for pocket knuckle obstructions
        for cx, cy, cr in self.main_sim.circles:
            # The moving path is the ball, the obstacle is the knuckle's radius
            if check_path_obstruction(
                    start_x, start_y, end_x, end_y,
                    cx, cy,
                    cr, OBJECT_BALL_RADIUS
            ):
                return True

        return False

    def is_cb_path_blocked(self, start_pos, end_pos, ignore_indices):
        """
        Checks if the straight-line path of the cue ball to a ghost ball
        is blocked by any other ball on the table or by pocket knuckles.
        """
        start_x, start_y = start_pos[0], start_pos[1]
        end_x, end_y = end_pos[0], end_pos[1]

        # Check for ball obstructions
        for i in range(1, self.main_sim.n_obj_balls + 1):
            if i in ignore_indices or not self.main_sim.in_play[i]:
                continue

            b_pos = self.main_sim.positions[i]

            if check_path_obstruction(
                    start_x, start_y, end_x, end_y,
                    b_pos[0], b_pos[1],
                    OBJECT_BALL_RADIUS, CUE_BALL_RADIUS
            ):
                return True

        # Check for pocket knuckle obstructions
        for cx, cy, cr in self.main_sim.circles:
            if check_path_obstruction(
                    start_x, start_y, end_x, end_y,
                    cx, cy,
                    cr, CUE_BALL_RADIUS
            ):
                return True

        return False

    def _is_path_blocked(self, start_pos, end_pos, moving_radius, ignore_indices):
        """
        Uses the Numba solver to check if a straight line is obstructed.
        """
        start_x, start_y = start_pos[0], start_pos[1]
        end_x, end_y = end_pos[0], end_pos[1]

        # Check for ball obstructions
        for i in range(1, self.main_sim.n_obj_balls + 1):
            if i in ignore_indices or not self.main_sim.in_play[i]:
                continue

            b_pos = self.main_sim.positions[i]
            if check_path_obstruction(
                    start_x, start_y, end_x, end_y,
                    b_pos[0], b_pos[1],
                    OBJECT_BALL_RADIUS, moving_radius
            ):
                return True

        # 2. Check for pocket knuckle obstructions
        for cx, cy, cr in self.main_sim.circles:
            if check_path_obstruction(
                    start_x, start_y, end_x, end_y,
                    cx, cy,
                    cr * 0.8, moving_radius
            ):
                return True

        return False

    def get_bank_pots(self):
        positions = self.main_sim.positions
        cue_ball_pos = positions[0]
        pocket_targets_list = self.get_pocket_targets()

        R_eff = np.sqrt(OBJECT_BALL_RADIUS ** 2 - (CUSHION_HEIGHT_EFF - OBJECT_BALL_RADIUS) ** 2)
        w = (TABLE_WIDTH / 2) - R_eff
        h = (TABLE_HEIGHT / 2) - R_eff

        cushions = ['left', 'right', 'top', 'bottom']
        bank_shots = []

        for target_idx in self.target_set:
            target_ball_pos = positions[target_idx]

            for pocket_idx, pocket_pts in enumerate(pocket_targets_list):
                for pt_idx, target_pt in enumerate(pocket_pts):
                    for cush in cushions:
                        mirrored_target = _mirror_point(target_pt, cush, w, h)
                        intersect_res = _get_intersection(target_ball_pos, mirrored_target, cush, w, h)

                        if not intersect_res:
                            continue

                        bounce_pt, t = intersect_res

                        if t <= 0 or not _in_bounds(bounce_pt, cush, w, h):
                            continue

                        dx_pot = bounce_pt[0] - target_ball_pos[0]
                        dy_pot = bounce_pt[1] - target_ball_pos[1]
                        dist_pot = np.hypot(dx_pot, dy_pot)

                        if dist_pot == 0: continue

                        dir_x_pot = dx_pot / dist_pot
                        dir_y_pot = dy_pot / dist_pot

                        gb_x = target_ball_pos[0] - (dir_x_pot * self.ghost_distance)
                        gb_y = target_ball_pos[1] - (dir_y_pot * self.ghost_distance)

                        dx_aim = gb_x - cue_ball_pos[0]
                        dy_aim = gb_y - cue_ball_pos[1]

                        bank_shots.append({
                            "target_idx": target_idx,
                            "pocket_idx": pocket_idx,
                            "target_pt_idx": pt_idx,
                            "ghost_ball_pos": (gb_x, gb_y),
                            "seed_angle": np.arctan2(dy_aim, dx_aim),
                            "target_pt": target_pt,
                            "type": f"1_cushion_bank_{cush}",
                            "bounce_points": [bounce_pt]
                        })

        return bank_shots

    def get_kick_pots(self, direct_shots, max_cushions=4):
        positions = self.main_sim.positions
        cue_ball_pos = positions[0]

        R_eff = np.sqrt(CUE_BALL_RADIUS ** 2 - (CUSHION_HEIGHT_EFF - CUE_BALL_RADIUS) ** 2)
        w = (TABLE_WIDTH / 2) - R_eff
        h = (TABLE_HEIGHT / 2) - R_eff

        cushions = ['left', 'right', 'top', 'bottom']

        sequences = [[c] for c in cushions]
        all_seqs = list(sequences)
        for _ in range(1, max_cushions):
            new_seqs = []
            for seq in sequences:
                for c in cushions:
                    if c != seq[-1]:
                        new_seqs.append(seq + [c])
            all_seqs.extend(new_seqs)
            sequences = new_seqs

        kick_shots = []

        for shot in direct_shots:
            ghost_pos = shot["ghost_ball_pos"]

            for seq in all_seqs:
                targets = [ghost_pos]
                for cush in reversed(seq):
                    targets.insert(0, _mirror_point(targets[0], cush, w, h))

                current_pos = cue_ball_pos
                bounce_points = []
                valid = True

                for i, cush in enumerate(seq):
                    aim_pt = targets[i]
                    intersect_res = _get_intersection(current_pos, aim_pt, cush, w, h)

                    if not intersect_res:
                        valid = False
                        break

                    bounce_pt, t = intersect_res

                    if t <= 0 or not _in_bounds(bounce_pt, cush, w, h):
                        valid = False
                        break

                    bounce_points.append(bounce_pt)
                    current_pos = bounce_pt

                if valid:
                    dx_aim = bounce_points[0][0] - cue_ball_pos[0]
                    dy_aim = bounce_points[0][1] - cue_ball_pos[1]
                    aim_angle = np.arctan2(dy_aim, dx_aim)

                    kick_shot = shot.copy()
                    kick_shot["type"] = f"{len(seq)}_cushion_kick"
                    kick_shot["sequence"] = seq
                    kick_shot["seed_angle"] = aim_angle
                    kick_shot["bounce_points"] = bounce_points

                    kick_shots.append(kick_shot)

        return kick_shots

    def get_plant_pots(self, min_efficiency=0.15, max_cut_angle_deg=80):
        """Calculates 2-ball plant (combination) shots with momentum, cut angle, and path culling."""
        positions = self.main_sim.positions
        cue_ball_pos = positions[0]
        pocket_targets_list = self.get_pocket_targets()

        # Convert max angle to its cosine equivalent for dot product comparison
        min_cos_cut = np.cos(np.radians(max_cut_angle_deg))
        plant_shots = []

        for target_idx in self.target_set:
            target_pos = positions[target_idx]

            for pocket_idx, pocket_pts in enumerate(pocket_targets_list):
                for pt_idx, target_pt in enumerate(pocket_pts):

                    if self._is_path_blocked(target_pos, target_pt, OBJECT_BALL_RADIUS, ignore_indices=[target_idx]):
                        continue

                    # Target Ball Potting Vector
                    dx_pot = target_pt[0] - target_pos[0]
                    dy_pot = target_pt[1] - target_pos[1]
                    dist_pot = np.hypot(dx_pot, dy_pot)
                    if dist_pot == 0: continue
                    dir_pot_x = dx_pot / dist_pot
                    dir_pot_y = dy_pot / dist_pot

                    # GB1: Where the Combo Ball must impact the Target Ball
                    gb1_x = target_pos[0] - (dir_pot_x * self.ghost_distance)
                    gb1_y = target_pos[1] - (dir_pot_y * self.ghost_distance)
                    gb1_pos = (gb1_x, gb1_y)

                    # Iterate through all other balls to act as the Combo Ball
                    for combo_idx in self.target_set:
                        if combo_idx == target_idx or not self.main_sim.in_play[combo_idx]:
                            continue

                        combo_pos = positions[combo_idx]

                        # PATH CULLING 2: Combo Ball to GB1
                        # Ignore the Target Ball (target_idx) as it is the destination of this path
                        if self._is_path_blocked(combo_pos, gb1_pos, OBJECT_BALL_RADIUS,
                                                 ignore_indices=[combo_idx, target_idx]):
                            continue

                        # Combo Ball to GB1 Vector
                        dx_c_gb1 = gb1_x - combo_pos[0]
                        dy_c_gb1 = gb1_y - combo_pos[1]
                        dist_c_gb1 = np.hypot(dx_c_gb1, dy_c_gb1)
                        if dist_c_gb1 == 0: continue
                        dir_c_gb1_x = dx_c_gb1 / dist_c_gb1
                        dir_c_gb1_y = dy_c_gb1 / dist_c_gb1

                        # Dot product = cos(theta) of the cut angle
                        cos_theta1 = (dir_pot_x * dir_c_gb1_x) + (dir_pot_y * dir_c_gb1_y)
                        if cos_theta1 < min_cos_cut:
                            continue

                        # GB2: Where the Cue Ball must impact the Combo Ball
                        gb2_x = combo_pos[0] - (dir_c_gb1_x * self.ghost_distance)
                        gb2_y = combo_pos[1] - (dir_c_gb1_y * self.ghost_distance)
                        gb2_pos = (gb2_x, gb2_y)

                        # PATH CULLING 3: Cue Ball to GB2
                        # Use CUE_BALL_RADIUS * 0.85 for swerve allowance
                        if self._is_path_blocked(cue_ball_pos, gb2_pos, CUE_BALL_RADIUS * 0.85,
                                                 ignore_indices=[0, combo_idx]):
                            continue

                        # Cue Ball to GB2 Vector
                        dx_cb_gb2 = gb2_x - cue_ball_pos[0]
                        dy_cb_gb2 = gb2_y - cue_ball_pos[1]
                        dist_cb_gb2 = np.hypot(dx_cb_gb2, dy_cb_gb2)
                        if dist_cb_gb2 == 0: continue
                        dir_cb_gb2_x = dx_cb_gb2 / dist_cb_gb2
                        dir_cb_gb2_y = dy_cb_gb2 / dist_cb_gb2

                        cos_theta2 = (dir_c_gb1_x * dir_cb_gb2_x) + (dir_c_gb1_y * dir_cb_gb2_y)
                        if cos_theta2 < min_cos_cut:
                            continue

                        # Multiply the transfer fractions to get the total momentum efficiency
                        efficiency = cos_theta1 * cos_theta2
                        if efficiency < min_efficiency:
                            continue

                        # Starter Aiming Angle
                        aim_angle = np.arctan2(dy_cb_gb2, dx_cb_gb2)

                        plant_shots.append({
                            "target_idx": target_idx,
                            "combo_idx": combo_idx,
                            "pocket_idx": pocket_idx,
                            "target_pt_idx": pt_idx,
                            "ghost_ball_pos": gb2_pos,  # CB aims at GB2
                            "gb1_pos": gb1_pos,  # Stored for visualizer rendering
                            "seed_angle": aim_angle,
                            "target_pt": target_pt,
                            "type": "plant",
                            "efficiency": efficiency,
                            "eff1": cos_theta1,
                            "eff2": cos_theta2
                        })

        return plant_shots

    def get_carom_pots(self, min_efficiency=0.1):
        """Calculates 2-ball carom (kiss) shots using right-triangle tangent math."""
        positions = self.main_sim.positions
        cue_ball_pos = positions[0]
        pocket_targets_list = self.get_pocket_targets()

        # At impact, the centers are 2 * radius apart
        R2 = OBJECT_BALL_RADIUS * 2
        carom_shots = []

        for target_idx in self.target_set:
            target_pos = positions[target_idx]

            # Iterate through all possible kiss balls
            for kiss_idx in range(1, self.main_sim.n_obj_balls + 1):
                if kiss_idx == target_idx or not self.main_sim.in_play[kiss_idx]:
                    continue

                kiss_pos = positions[kiss_idx]

                for pocket_idx, pocket_pts in enumerate(pocket_targets_list):
                    for pt_idx, target_pt in enumerate(pocket_pts):

                        dx_kp = target_pt[0] - kiss_pos[0]
                        dy_kp = target_pt[1] - kiss_pos[1]
                        dist_kp = np.hypot(dx_kp, dy_kp)

                        # If the pocket is physically inside the kiss ball, impossible
                        if dist_kp <= R2:
                            continue

                        # The angle of the right triangle formed by the collision
                        theta = np.arccos(R2 / dist_kp)
                        alpha = np.arctan2(dy_kp, dx_kp)

                        # There are always two valid sides to kiss off a ball
                        for sign in [-1, 1]:
                            beta = alpha + (sign * theta)

                            # Impact Point (GB1): Where the Target Ball must be at the moment of collision
                            ix = kiss_pos[0] + R2 * np.cos(beta)
                            iy = kiss_pos[1] + R2 * np.sin(beta)

                            # Vector 1: Target Ball to Impact Point
                            dx_ti = ix - target_pos[0]
                            dy_ti = iy - target_pos[1]
                            dist_ti = np.hypot(dx_ti, dy_ti)
                            if dist_ti == 0: continue

                            dir_ti_x = dx_ti / dist_ti
                            dir_ti_y = dy_ti / dist_ti

                            # Normal Vector: Impact Point to Kiss Ball
                            nx = (kiss_pos[0] - ix) / R2
                            ny = (kiss_pos[1] - iy) / R2

                            # CULL 1: The Target Ball must actually hit the Kiss Ball, not back-cut it
                            cos_cut1 = (dir_ti_x * nx) + (dir_ti_y * ny)
                            if cos_cut1 <= 0.01:
                                continue

                                # Tangent Vector: Impact Point to Pocket
                            dist_ip = np.hypot(target_pt[0] - ix, target_pt[1] - iy)
                            px_dir = (target_pt[0] - ix) / dist_ip
                            py_dir = (target_pt[1] - iy) / dist_ip

                            # CULL 2: The Target Ball must deflect FORWARD toward the pocket.
                            # The sine of the cut angle represents the momentum preserved along the tangent.
                            sin_cut1 = (dir_ti_x * px_dir) + (dir_ti_y * py_dir)
                            if sin_cut1 <= 0.01:
                                continue

                                # GB2: Where the Cue Ball must impact the Target Ball
                            gb2_x = target_pos[0] - (dir_ti_x * self.ghost_distance)
                            gb2_y = target_pos[1] - (dir_ti_y * self.ghost_distance)

                            # Vector 2: Cue Ball to GB2
                            dx_cb = gb2_x - cue_ball_pos[0]
                            dy_cb = gb2_y - cue_ball_pos[1]
                            dist_cb = np.hypot(dx_cb, dy_cb)
                            if dist_cb == 0: continue

                            dir_cb_x = dx_cb / dist_cb
                            dir_cb_y = dy_cb / dist_cb

                            # CULL 3: The Cue Ball must not back-cut the Target Ball
                            cos_cut2 = (dir_cb_x * dir_ti_x) + (dir_cb_y * dir_ti_y)
                            if cos_cut2 <= 0.01:
                                continue

                            # CULL 4: Momentum Transfer Efficiency
                            efficiency = sin_cut1 * cos_cut2
                            if efficiency < min_efficiency:
                                continue

                            aim_angle = np.arctan2(dy_cb, dx_cb)

                            carom_shots.append({
                                "target_idx": target_idx,
                                "kiss_idx": kiss_idx,
                                "pocket_idx": pocket_idx,
                                "target_pt_idx": pt_idx,
                                "ghost_ball_pos": (gb2_x, gb2_y),
                                "gb1_pos": (ix, iy),
                                "seed_angle": aim_angle,
                                "target_pt": target_pt,
                                "type": "carom",
                                "efficiency": efficiency
                            })

        return carom_shots

    def get_cue_carom_pots(self, min_efficiency=0.1):
        """Calculates cue ball caroms (cannons) where the CB hits a legal ball and deflects into a target ball."""
        positions = self.main_sim.positions
        cue_ball_pos = positions[0]
        pocket_targets_list = self.get_pocket_targets()

        # At impact, the centers are CB radius + OB radius apart
        R2 = OBJECT_BALL_RADIUS + CUE_BALL_RADIUS
        cue_carom_shots = []

        # The Target Ball is the one actually going into the pocket (can be any ball)
        for target_idx in range(1, self.main_sim.n_obj_balls + 1):
            if not self.main_sim.in_play[target_idx]:
                continue

            target_pos = positions[target_idx]

            for pocket_idx, pocket_pts in enumerate(pocket_targets_list):
                for pt_idx, target_pt in enumerate(pocket_pts):

                    dx_pot = target_pt[0] - target_pos[0]
                    dy_pot = target_pt[1] - target_pos[1]
                    dist_pot = np.hypot(dx_pot, dy_pot)
                    if dist_pot == 0: continue

                    dir_pot_x = dx_pot / dist_pot
                    dir_pot_y = dy_pot / dist_pot

                    # GB2: Where the CB must be to pot the Target Ball
                    gb2_x = target_pos[0] - (dir_pot_x * self.ghost_distance)
                    gb2_y = target_pos[1] - (dir_pot_y * self.ghost_distance)

                    # OB1 is the first ball the CB hits. Must be a legal target.
                    for ob1_idx in self.target_set:
                        if ob1_idx == target_idx or not self.main_sim.in_play[ob1_idx]:
                            continue

                        if self._is_ob_path_blocked(positions[target_idx], target_pt,
                                                    ignore_indices=[0, target_idx, ob1_idx]):
                            continue

                        ob1_pos = positions[ob1_idx]

                        dx_ob1_gb2 = gb2_x - ob1_pos[0]
                        dy_ob1_gb2 = gb2_y - ob1_pos[1]
                        dist_ob1_gb2 = np.hypot(dx_ob1_gb2, dy_ob1_gb2)

                        # If the required CB position overlaps the first object ball, impossible
                        if dist_ob1_gb2 <= R2:
                            continue

                        # Right triangle math to find the 90-degree tangent departure points
                        theta = np.arccos(R2 / dist_ob1_gb2)
                        alpha = np.arctan2(dy_ob1_gb2, dx_ob1_gb2)

                        for sign in [-1, 1]:
                            beta = alpha + (sign * theta)

                            # GB1: Where the CB must impact OB1
                            gb1_x = ob1_pos[0] + R2 * np.cos(beta)
                            gb1_y = ob1_pos[1] + R2 * np.sin(beta)

                            # Vector 1: Cue Ball to GB1
                            dx_cb_gb1 = gb1_x - cue_ball_pos[0]
                            dy_cb_gb1 = gb1_y - cue_ball_pos[1]
                            dist_cb_gb1 = np.hypot(dx_cb_gb1, dy_cb_gb1)
                            if dist_cb_gb1 == 0: continue

                            dir_cb_x = dx_cb_gb1 / dist_cb_gb1
                            dir_cb_y = dy_cb_gb1 / dist_cb_gb1

                            # Normal Vector at OB1 impact
                            nx = (ob1_pos[0] - gb1_x) / R2
                            ny = (ob1_pos[1] - gb1_y) / R2

                            # CULL 1: CB must hit OB1, not back-cut it
                            cos_cut1 = (dir_cb_x * nx) + (dir_cb_y * ny)
                            if cos_cut1 <= 0.01:
                                continue

                            # Tangent Vector: GB1 to GB2
                            dist_gb1_gb2 = np.hypot(gb2_x - gb1_x, gb2_y - gb1_y)
                            tan_dir_x = (gb2_x - gb1_x) / dist_gb1_gb2
                            tan_dir_y = (gb2_y - gb1_y) / dist_gb1_gb2

                            # The momentum retained by the CB along the tangent is proportional to the sine of the cut
                            sin_cut1 = (dir_cb_x * tan_dir_x) + (dir_cb_y * tan_dir_y)
                            if sin_cut1 <= 0.01:
                                continue

                            # CULL 2: CB must not back-cut the Target Ball at GB2
                            cos_cut2 = (tan_dir_x * dir_pot_x) + (tan_dir_y * dir_pot_y)
                            if cos_cut2 <= 0.01:
                                continue

                            efficiency = sin_cut1 * cos_cut2
                            if efficiency < min_efficiency:
                                continue

                            aim_angle = np.arctan2(dy_cb_gb1, dx_cb_gb1)

                            cue_carom_shots.append({
                                "target_idx": target_idx,
                                "ob1_idx": ob1_idx,
                                "pocket_idx": pocket_idx,
                                "target_pt_idx": pt_idx,
                                "ghost_ball_pos": (gb1_x, gb1_y),  # CB aims here first
                                "gb2_pos": (gb2_x, gb2_y),  # Where CB hits Target
                                "seed_angle": aim_angle,
                                "target_pt": target_pt,
                                "type": "cue_carom",
                                "efficiency": efficiency
                            })

        return cue_carom_shots

    def get_all_shots(self, max_kick_cushions=0):
        direct = self.get_direct_pots()
        plants = self.get_plant_pots()

        # Only take highly efficient plants to save time
        viable_plants = [shot for shot in plants if shot["efficiency"] > 0.7]

        # Ignore kicks and banks entirely for now
        return direct + viable_plants

    def get_safety_candidates(self, num_angles=15):
        """Generates a spread of ghost balls around the face of all legal target balls."""
        positions = self.main_sim.positions
        cue_ball_pos = positions[0]
        safety_candidates = []

        # Sweep from -80 degrees to +80 degrees to cover thin cuts to full-ball
        sweep_angles = np.linspace(-np.radians(80), np.radians(80), num_angles)

        for target_idx in self.target_set:
            target_pos = positions[target_idx]

            # Base vector from Cue Ball to Target Ball
            dx = target_pos[0] - cue_ball_pos[0]
            dy = target_pos[1] - cue_ball_pos[1]
            dist = np.hypot(dx, dy)

            if dist == 0:
                continue

            base_angle = np.arctan2(dy, dx)

            for offset in sweep_angles:
                # The angle from the center of the target ball out to the ghost ball
                impact_angle = base_angle + np.pi + offset

                # Calculate the exact Ghost Ball position touching the target ball
                gb_x = target_pos[0] + np.cos(impact_angle) * self.ghost_distance
                gb_y = target_pos[1] + np.sin(impact_angle) * self.ghost_distance

                # Cull the shot if the straight line to the Ghost Ball is blocked
                if self.is_cb_path_blocked(cue_ball_pos, (gb_x, gb_y), ignore_indices=[0, target_idx]):
                    continue

                # The aim vector from the cue ball to this specific ghost ball
                dx_aim = gb_x - cue_ball_pos[0]
                dy_aim = gb_y - cue_ball_pos[1]

                safety_candidates.append({
                    "target_idx": target_idx,
                    "ghost_ball_pos": (gb_x, gb_y),
                    "seed_angle": np.arctan2(dy_aim, dx_aim),
                    "type": "safety"
                })

        return safety_candidates

    def get_1_cushion_escapes(self):
        """Generates fast 1-cushion kick shots to escape snookers."""
        positions = self.main_sim.positions
        cue_ball_pos = positions[0]

        R_eff = np.sqrt(CUE_BALL_RADIUS ** 2 - (CUSHION_HEIGHT_EFF - CUE_BALL_RADIUS) ** 2)
        w = (TABLE_WIDTH / 2) - R_eff
        h = (TABLE_HEIGHT / 2) - R_eff
        cushions = ['left', 'right', 'top', 'bottom']

        escapes = []

        for target_idx in self.target_set:
            target_pos = positions[target_idx]

            for cush in cushions:
                mirrored_target = _mirror_point(target_pos, cush, w, h)
                intersect_res = _get_intersection(cue_ball_pos, mirrored_target, cush, w, h)

                if not intersect_res:
                    continue

                bounce_pt, t = intersect_res

                if t <= 0 or not _in_bounds(bounce_pt, cush, w, h):
                    continue

                # Cull if path from Cue Ball to Cushion is blocked
                if self.is_cb_path_blocked(cue_ball_pos, bounce_pt, ignore_indices=[0]):
                    continue
                # Cull if path from Cushion to Target Ball is blocked
                if self.is_cb_path_blocked(bounce_pt, target_pos, ignore_indices=[0, target_idx]):
                    continue

                dx_aim = bounce_pt[0] - cue_ball_pos[0]
                dy_aim = bounce_pt[1] - cue_ball_pos[1]

                escapes.append({
                    "target_idx": target_idx,
                    "ghost_ball_pos": target_pos,
                    "seed_angle": np.arctan2(dy_aim, dx_aim),
                    "type": "1_cushion_escape"
                })

        return escapes