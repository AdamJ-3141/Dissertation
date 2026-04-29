import numpy as np
from pool_simulation.constants import *
import json
import os
from .aim_solver import check_escape_rays_numba


def dist_to_segment(start, end, point):
    line_vec = end - start
    point_vec = point - start
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq == 0: return np.linalg.norm(point_vec)
    t = np.clip(np.dot(point_vec, line_vec) / line_len_sq, 0, 1)
    projection = start + t * line_vec
    return np.linalg.norm(point - projection)


class TableEvaluator:
    def __init__(self, sim, target_colour=1, weights=None):
        self.sim = sim
        self.colour_set = target_colour

        # Load absolute defaults from JSON
        path = os.path.join(os.path.dirname(__file__), 'defaults.json')
        with open(path, 'r') as f:
            all_weights = json.load(f)

        # Overwrite with any custom agent weights
        if weights:
            all_weights.update(weights)

        self.w = all_weights
        self.pocket_targets = self._precompute_pocket_targets(points_per_pocket=7)

    def _precompute_pocket_targets(self, points_per_pocket=7, spread_ratio=0.65):
        target_points = []
        for p_info in self.sim.pockets:
            px, py, r = p_info[0], p_info[1], p_info[2]

            vx = 0.0 if abs(px) < 0.01 else -np.sign(px)
            vy = 0.0 if abs(py) < 0.01 else -np.sign(py)

            length = np.hypot(vx, vy)
            if length > 0:
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
                pocket_pts.append(np.array([mouth_x + (tx * offset), mouth_y + (ty * offset)]))
            target_points.append(pocket_pts)

        return target_points

    def is_path_clear(self, start_pos, end_pos, ignore_indices=None):
        if ignore_indices is None: ignore_indices = []

        p1 = start_pos[:2]
        p2 = end_pos[:2]

        v_path = p2 - p1
        dist_path = np.linalg.norm(v_path)
        if dist_path == 0: return True
        dir_path = v_path / dist_path

        # 1. Check for ball obstructions
        for i in range(1, self.sim.n_obj_balls + 1):
            if i in ignore_indices or not self.sim.in_play[i]: continue
            v_ball = self.sim.positions[i][:2] - p1
            projection = np.dot(v_ball, dir_path)

            if 0 < projection < dist_path:
                dist_to_line = np.linalg.norm(v_ball - projection * dir_path)
                if dist_to_line < (OBJECT_BALL_RADIUS * 2.0):
                    return False

        # 2. Check for knuckle obstructions
        for cx, cy, cr in self.sim.circles:
            if cr < 0.2:  # Only evaluate actual knuckles, skip massive cushion bounds
                v_circ = np.array([cx, cy]) - p1
                projection = np.dot(v_circ, dir_path)

                if 0 < projection < dist_path:
                    dist_to_line = np.linalg.norm(v_circ - projection * dir_path)
                    if dist_to_line < (cr + OBJECT_BALL_RADIUS):
                        return False

        return True

    def get_pocket_access_multiplier(self, t_idx, p_idx):
        t_pos = self.sim.positions[t_idx][:2]
        target_pts = self.pocket_targets[p_idx]

        visible_count = 0
        for pt in target_pts:
            if self.is_path_clear(t_pos, pt, ignore_indices=[t_idx]):
                visible_count += 1

        return visible_count / len(target_pts)

    def get_single_ball_heatmap(self, target_idx, nx=100, ny=50):
        playable_w, playable_h = (TABLE_WIDTH / 2) - CUE_BALL_RADIUS, (TABLE_HEIGHT / 2) - CUE_BALL_RADIUS
        x, y = np.linspace(-playable_w, playable_w, nx), np.linspace(-playable_h, playable_h, ny)
        X, Y = np.meshgrid(x, y)
        final_heatmap = np.zeros((ny, nx))
        if not self.sim.in_play[target_idx]: return final_heatmap, playable_w, playable_h

        t_pos, R = self.sim.positions[target_idx], OBJECT_BALL_RADIUS

        obstacles = []
        for i in range(1, self.sim.n_obj_balls + 1):
            if self.sim.in_play[i] and i != target_idx:
                obstacles.append((self.sim.positions[i][0], self.sim.positions[i][1], OBJECT_BALL_RADIUS))
        for cx, cy, cr in self.sim.circles:
            obstacles.append((cx, cy, cr))

        for p_idx, p_info in enumerate(self.sim.pockets):
            p_pos = p_info[:2]
            v_tp = p_pos - t_pos
            norm_tp = np.linalg.norm(v_tp)
            if norm_tp == 0: continue
            dir_tp = v_tp / norm_tp

            raw_normal = np.array([p_pos[0], p_pos[1]])
            pocket_normal = raw_normal / np.linalg.norm(raw_normal)
            if np.dot(dir_tp, pocket_normal) < 0.05: continue

            # Calculate Exact Ghost Ball Coordinate
            gb_x = t_pos[0] - (2 * R) * dir_tp[0]
            gb_y = t_pos[1] - (2 * R) * dir_tp[1]

            v_gt_X, v_gt_Y = gb_x - X, gb_y - Y
            norm_gt = np.hypot(v_gt_X, v_gt_Y)
            norm_gt[norm_gt == 0] = 1e-8

            dist_score = np.ones_like(norm_gt)
            dist_score[norm_gt < 0.15] = np.exp(
                -0.5 * ((norm_gt[norm_gt < 0.15] - 0.15) / self.w["close_plateau_sigma"]) ** 2)
            dist_score[norm_gt > 0.40] = np.exp(
                -0.5 * ((norm_gt[norm_gt > 0.40] - 0.40) / self.w["far_plateau_sigma"]) ** 2)

            cos_theta = (dir_tp[0] * (v_gt_X / norm_gt)) + (dir_tp[1] * (v_gt_Y / norm_gt))
            angle_score = np.ones_like(cos_theta)
            angle_score[cos_theta > 0.965] = np.exp(
                -0.5 * ((cos_theta[cos_theta > 0.965] - 0.965) / self.w["straight_penalty_sigma"]) ** 2)

            proximity = np.clip((0.7 - norm_tp) / 0.7, 0.0, 1.0)
            base_thresh = (0.82 - (proximity * 0.2)) if p_idx in [1, 4] else (0.76 - (proximity * 0.26))
            dyn_thresh = np.clip(base_thresh + (np.maximum(0.0, norm_gt - 0.4) * 0.50), base_thresh, 0.96)

            angle_score[cos_theta < dyn_thresh] = np.exp(-0.5 * (
                        (cos_theta[cos_theta < dyn_thresh] - dyn_thresh[cos_theta < dyn_thresh]) / self.w[
                    "thin_penalty_sigma"]) ** 2)
            angle_score[cos_theta <= 0.0] = 0.0

            visibility = self.get_pocket_access_multiplier(target_idx, p_idx)
            difficulty_mod = np.clip(visibility * np.exp(-0.5 * (norm_tp / self.w["dist_decay_sigma"]) ** 2), 0.4, 1.0)

            # Base pocket heatmap
            pocket_heatmap = (dist_score * angle_score) ** self.w["pos_quality_exponent"] * difficulty_mod

            dx = gb_x - X
            dy = gb_y - Y
            line_len_sq = dx ** 2 + dy ** 2
            line_len_sq[line_len_sq < 1e-8] = 1e-8  # Prevent division by zero

            for ox, oy, obs_radius in obstacles:
                # Project obstacle center onto the path vector for the entire grid
                t = ((ox - X) * dx + (oy - Y) * dy) / line_len_sq
                t = np.clip(t, 0.0, 1.0)

                closest_x = X + t * dx
                closest_y = Y + t * dy

                dist_sq = (ox - closest_x) ** 2 + (oy - closest_y) ** 2

                # Zero out any pixel where the cue ball path clips the obstacle
                pocket_heatmap[dist_sq < (obs_radius + CUE_BALL_RADIUS) ** 2] = 0.0

            final_heatmap = np.maximum(final_heatmap, pocket_heatmap)

        return np.clip(final_heatmap, 0.0, 1.0), playable_w, playable_h

    def get_full_heatmap(self, nx=100, ny=50):
        playable_w, playable_h = (TABLE_WIDTH / 2) - CUE_BALL_RADIUS, (TABLE_HEIGHT / 2) - CUE_BALL_RADIUS
        total_traffic = np.zeros((ny, nx))

        targets = [i for i in range(1, self.sim.n_obj_balls + 1) if
                   self.sim.colours[i] in (self.colour_set, 3) and self.sim.in_play[i]]

        if not targets: return total_traffic, playable_w, playable_h, 0.0

        for t_idx in targets:
            ball_heatmap, _, _ = self.get_single_ball_heatmap(t_idx, nx, ny)

            if self.sim.colours[t_idx] == 3:
                ball_heatmap *= self.w.get("black_weight", 0.2)

            total_traffic = np.maximum(total_traffic, ball_heatmap)

        max_val = max(1e-8, total_traffic.max())
        normalized = total_traffic / max_val

        freedom_score = np.mean(normalized)

        return normalized, playable_w, playable_h, freedom_score

    def direct_pots_score(self):
        targets = [i for i in range(1, self.sim.n_obj_balls + 1) if
                   self.sim.colours[i] in (self.colour_set, 3) and self.sim.in_play[i]]
        cb_pos, total_score = self.sim.positions[0], 0.0

        for t_idx in targets:
            t_pos = self.sim.positions[t_idx]
            pocket_scores = []

            for p_idx in range(len(self.sim.pockets)):
                visibility = self.get_pocket_access_multiplier(t_idx, p_idx)
                if visibility <= 0: continue

                p_pos = self.sim.pockets[p_idx][:2]

                v_tp = p_pos - t_pos
                dist_tp = np.linalg.norm(v_tp)
                if dist_tp < 1e-6: continue
                dir_tp = v_tp / dist_tp

                gb_pos = t_pos - (2 * OBJECT_BALL_RADIUS) * dir_tp
                if not self.is_path_clear(cb_pos, gb_pos, [0, t_idx]): continue

                v_cb_gb = gb_pos - cb_pos
                dist_cb_gb = np.linalg.norm(v_cb_gb)
                if dist_cb_gb < 1e-6: continue
                dir_cb_gb = v_cb_gb / dist_cb_gb

                cos_theta = np.dot(dir_cb_gb, dir_tp)
                if cos_theta <= 0.0: continue

                ob_dist_score = np.exp(-0.5 * (dist_tp / (self.w["dist_decay_sigma"] * 0.4)) ** 2)
                cb_dist_score = np.exp(-0.5 * (dist_cb_gb / self.w["dist_decay_sigma"]) ** 2)

                forgiveness = np.clip((0.15 - dist_tp) / 0.15, 0.0, 1.0)

                base_cut_score = cos_theta ** 2
                dynamic_cut = base_cut_score + (1.0 - base_cut_score) * forgiveness

                shot_val = ob_dist_score * cb_dist_score * dynamic_cut * visibility

                if self.sim.colours[t_idx] == 3:
                    shot_val *= self.w.get("black_weight", 0.1)

                # Only keep reasonably possible pots
                if shot_val > 0.05:
                    pocket_scores.append(shot_val)

            if pocket_scores:
                # Sort descending so the best shot is always index 0
                pocket_scores.sort(reverse=True)

                target_score = pocket_scores[0]

                # Add secondary options as fractional "insurance" bonuses
                if len(pocket_scores) > 1:
                    target_score += pocket_scores[1] * 0.25  # 2nd pocket is worth 25%
                if len(pocket_scores) > 2:
                    target_score += pocket_scores[2] * 0.10  # 3rd pocket is worth 10%

                total_score += target_score

        return total_score

    def visibility_analysis_score(self):
        targets = [i for i in range(1, self.sim.n_obj_balls + 1) if
                   self.sim.colours[i] in (self.colour_set, 3) and self.sim.in_play[i]]

        if not targets:
            return 0.0

        cb_pos = self.sim.positions[0][:2]
        total_visibility_points = 0

        for t_idx in targets:
            t_pos = self.sim.positions[t_idx][:2]
            v_cb_t = t_pos - cb_pos
            dist = np.linalg.norm(v_cb_t)

            if dist < 1e-6:
                total_visibility_points += 3
                continue

            dir_t = v_cb_t / dist
            # Get the perpendicular vector to find the extreme left and right edges
            perp = np.array([-dir_t[1], dir_t[0]])

            # The extreme edges the cue ball center can aim at to graze the object ball.
            # We use 1.95 * R instead of 2.0 to prevent floating-point micro-grazes from returning True.
            offset = perp * (OBJECT_BALL_RADIUS * 1.95)

            # Check all 3 physical lines of sight
            center_clear = self.is_path_clear(cb_pos, t_pos, ignore_indices=[0, t_idx])
            left_clear = self.is_path_clear(cb_pos, t_pos + offset, ignore_indices=[0, t_idx])
            right_clear = self.is_path_clear(cb_pos, t_pos - offset, ignore_indices=[0, t_idx])

            if center_clear: total_visibility_points += 1
            if left_clear: total_visibility_points += 1
            if right_clear: total_visibility_points += 1

        # Evaluate the penalty based on how many total "lines of sight" we have to our legal balls
        if total_visibility_points == 0:
            return -1.5  # Total Snooker (cannot hit ANY part of ANY legal ball)
        elif total_visibility_points <= 2:
            return -0.5  # Severe Snooker (can only scrape the edge of one ball)
        elif total_visibility_points < len(targets) * 2:
            return -0.2  # Minor restriction
        else:
            return 0.0  # We have sufficient visibility across the table, no penalty applied

    def cluster_analysis_score(self):
        targets = [i for i in range(1, self.sim.n_obj_balls + 1) if
                   self.sim.colours[i] in (self.colour_set, 3) and self.sim.in_play[i]]
        all_active = [i for i in range(1, self.sim.n_obj_balls + 1) if self.sim.in_play[i]]
        score, cluster_centers = 0.0, []
        thresh = OBJECT_BALL_RADIUS * self.w["cluster_dist_thresh"]

        for i in all_active:
            pos_i = self.sim.positions[i][:2]

            for j in all_active:
                if i >= j: continue
                pos_j = self.sim.positions[j][:2]
                if np.linalg.norm(pos_i - pos_j) < thresh:
                    # Package the coordinates and the ball indices
                    cluster_centers.append(((pos_i + pos_j) / 2, [i, j]))

            cushions = [
                np.array([pos_i[0], TABLE_HEIGHT / 2]),
                np.array([pos_i[0], -TABLE_HEIGHT / 2]),
                np.array([TABLE_WIDTH / 2, pos_i[1]]),
                np.array([-TABLE_WIDTH / 2, pos_i[1]])
            ]
            for c_pt in cushions:
                if np.linalg.norm(pos_i - c_pt) < thresh:
                    # Use a string tag for the cushion phantom ball
                    cluster_centers.append(((pos_i + c_pt) / 2, [i, "cushion"]))

        penalties = 0.0
        bonuses = 0.0

        for t_idx in targets:
            has_pot = any(self.get_pocket_access_multiplier(t_idx, p) > 0.4 for p in range(len(self.sim.pockets)))
            in_cluster = any(np.linalg.norm(self.sim.positions[t_idx][:2] - c_pos) < thresh for c_pos, members in cluster_centers)
            if in_cluster: score += self.w["trapped_penalty"] if not has_pot else self.w["congested_penalty"]
            if has_pot:
                for p_idx in range(len(self.sim.pockets)):
                    score += self._breakout_potential(t_idx, self.sim.pockets[p_idx][:2], cluster_centers)
            if in_cluster:
                penalties += self.w["trapped_penalty"] if not has_pot else self.w["congested_penalty"]

            if has_pot:
                for p_idx in range(len(self.sim.pockets)):
                    bonuses += self._breakout_potential(t_idx, self.sim.pockets[p_idx][:2], cluster_centers)

        # 1. If the table is completely open, it scores a perfect 0.0
        if penalties == 0.0:
            return 0.0

        # 2. If clusters exist, enforce Residual Risk.
        # Bonuses can only offset up to 85% of the total penalties.
        max_recovery = abs(penalties) * 0.85
        applied_bonus = min(bonuses, max_recovery)

        return (penalties + applied_bonus) / max(1, len(targets))

    def _breakout_potential(self, t_idx, p_pos, clusters):
        t_pos = self.sim.positions[t_idx][:2]
        dir_p = (p_pos - t_pos) / np.linalg.norm(p_pos - t_pos)
        gb_pos = t_pos - (2 * OBJECT_BALL_RADIUS) * dir_p
        tangent = np.array([-dir_p[1], dir_p[0]])
        val = 0.0

        for side in [-1, 1]:
            for c_pos, members in clusters:

                may_pot_black = False
                black_idx = next((m for m in members if m != "cushion" and self.sim.colours[m] == 3), None)

                if black_idx is not None:
                    b_pos = self.sim.positions[black_idx][:2]

                    # 1. Find the nearest pocket to the black ball
                    min_dist = float('inf')
                    nearest_pocket = None
                    for p in self.sim.pockets:
                        d = np.linalg.norm(b_pos - np.array(p[:2]))
                        if d < min_dist:
                            min_dist = d
                            nearest_pocket = np.array(p[:2])

                    # 2. Check if the black ball is dangerously close
                    if min_dist < 0.2:
                        black_is_closest = True

                        # 3. Check if its cluster partner shields it from the pocket
                        for m in members:
                            if m != "cushion" and m != black_idx:
                                m_pos = self.sim.positions[m][:2]
                                if np.linalg.norm(m_pos - nearest_pocket) < min_dist:
                                    black_is_closest = False
                                    break

                        if black_is_closest:
                            may_pot_black = True

                if may_pot_black:
                    continue

                dist = np.linalg.norm(c_pos - gb_pos)
                if 0 < dist < self.w["breakout_dist_max"] and np.dot((c_pos - gb_pos) / dist, tangent * side) > self.w[
                    "breakout_cone_angle"]:
                    val += (self.w["breakout_bonus_coeff"] / dist)

        return val

    def get_table_clearability_score(self):
        targets = [i for i in range(1, self.sim.n_obj_balls + 1) if
                   self.sim.colours[i] in (self.colour_set, 3) and self.sim.in_play[i]]
        if not targets: return 0.0

        _, _, _, freedom_easiness = self.get_full_heatmap()  # Already strictly 0.0 to 1.0

        # Normalize attackability to a maximum of ~1.0
        raw_attackability = self.direct_pots_score() / len(targets)
        theoretical_max_pot_score = 1.35
        norm_attackability = min(1.0, raw_attackability / theoretical_max_pot_score)

        norm_clusters = self.cluster_analysis_score()
        norm_visibility = self.visibility_analysis_score()

        w_easiness = self.w.get("w_easiness", 0.33)
        w_attackability = self.w.get("w_attackability", 0.33)
        w_strategic = self.w.get("w_strategic", 0.33)
        w_safety = self.w.get("w_safety", 0.15)

        # Force the GA's mutated weights to act as percentages of 100%
        total_w = w_easiness + w_attackability + abs(w_strategic) + abs(w_safety)
        if total_w > 0:
            w_easiness /= total_w
            w_attackability /= total_w
            w_strategic /= total_w
            w_safety /= total_w

        return (freedom_easiness * w_easiness) + \
            (norm_attackability * w_attackability) + \
            (norm_clusters * w_strategic) + \
            (norm_visibility * w_safety)

    def is_ghost_ball_accessible(self, gb_pos, ignore_indices=None):
        if ignore_indices is None: ignore_indices = []

        # 1. Out of Bounds Check
        playable_w = (TABLE_WIDTH / 2) - CUE_BALL_RADIUS
        playable_h = (TABLE_HEIGHT / 2) - CUE_BALL_RADIUS
        if not (-playable_w <= gb_pos[0] <= playable_w and -playable_h <= gb_pos[1] <= playable_h):
            return False

        # 2. Compile obstacles into a 2D Numba-friendly array
        obs_list = []
        for i in range(1, self.sim.n_obj_balls + 1):
            if i in ignore_indices or not self.sim.in_play[i]: continue

            # Instant overlap check while we loop
            dist = np.linalg.norm(self.sim.positions[i][:2] - gb_pos)
            if dist < (CUE_BALL_RADIUS + OBJECT_BALL_RADIUS - 1e-4):
                return False

            obs_list.append([self.sim.positions[i][0], self.sim.positions[i][1], OBJECT_BALL_RADIUS])

        for cx, cy, cr in self.sim.circles:
            if cr < 0.2:
                obs_list.append([cx, cy, cr])

        if not obs_list:
            return True

        obstacles_array = np.array(obs_list, dtype=np.float64)

        return check_escape_rays_numba(
            gb_pos[0], gb_pos[1],
            obstacles_array,
            num_rays=360,
            escape_dist=2.0,
            path_radius=CUE_BALL_RADIUS
        )