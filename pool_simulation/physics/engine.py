import numpy as np
from pool_simulation.constants import *


class Simulation:
    def __init__(self, n_balls=15,
                 cb_radius=CUE_BALL_RADIUS, cb_mass=CUE_BALL_MASS,
                 ob_radius=OBJECT_BALL_RADIUS, ob_mass=OBJECT_BALL_MASS,
                 mu_s=MU_S, mu_r=MU_R, dt_max=1.0 / 60):

        self.n_balls = n_balls
        self.table_width = TABLE_WIDTH
        self.table_height = TABLE_HEIGHT
        self.cb_mass = cb_mass
        self.cb_radius = cb_radius
        self.ob_mass = ob_mass
        self.ob_radius = ob_radius
        self.mu_s = mu_s  # sliding friction coefficient
        self.mu_r = mu_r  # rolling friction coefficient

        self.g = 9.81

        # State arrays
        self.positions = np.zeros((1 + n_balls, 2), dtype=np.float64)
        self.velocities = np.zeros((1 + n_balls, 2), dtype=np.float64)
        self.sliding_velocities = np.zeros((1 + n_balls, 2), dtype=np.float64)
        self.angular = np.zeros((1 + n_balls, 3), dtype=np.float64)
        self.radii = np.array([cb_radius] + [ob_radius] * n_balls, dtype=np.float64)
        self.in_play = np.ones(1 + n_balls, dtype=bool)
        self.colours = np.zeros(1 + n_balls, dtype=np.int8)
        self.ball_states = np.zeros(1 + n_balls, dtype=str)

        # internal simulation clock
        self.time = 0.0
        self.dt_max = dt_max

        # per-ball phase trackers
        self.phase = ["sliding"] * (1 + n_balls)  # or "rolling" or "stopped"
        self.next_event_time = np.zeros(1 + n_balls)

    def reset(self, positions=None, colours=None, in_play=None):
        """Reset to given positions/velocities, or random if None."""
        if positions is None:
            self.positions[:] = np.random.rand(self.n_balls, 2) * [
                self.table_width, self.table_height
            ]
        else:
            self.positions[:] = positions

        if colours is None:
            self.colours[:] = 0
            cutoff = np.random.randint(0, self.n_balls + 1)
            self.colours[cutoff:self.n_balls] = 1
            self.colours[-1] = 2
        else:
            self.colours[:] = colours

        if in_play is None:
            self.in_play = np.ones_like(self.colours, dtype=bool)
        else:
            self.in_play[:] = in_play

        self.angular.fill(0.0)
        self.velocities.fill(0.0)
        self.phase = ["sliding"] * (1 + self.n_balls)
        self.time = 0.0

    def reset_to_break(self):
        """
        Resets the table to break position
        :return:
        """
        try:
            assert self.n_balls == 15
        except AssertionError:
            raise AttributeError("Number of balls must be 15 to reset to break.")
        self.reset(
            np.array([
                [-0.547, 0],
                [BLACK_SPOT_X - 2 * SQRT_3_INCHES, 0],
                [BLACK_SPOT_X - SQRT_3_INCHES, INCHES_TO_M],
                [BLACK_SPOT_X, -2 * INCHES_TO_M],
                [BLACK_SPOT_X + SQRT_3_INCHES, 3 * INCHES_TO_M],
                [BLACK_SPOT_X + SQRT_3_INCHES, -1 * INCHES_TO_M],
                [BLACK_SPOT_X + 2 * SQRT_3_INCHES, 2 * INCHES_TO_M],
                [BLACK_SPOT_X + 2 * SQRT_3_INCHES, -4 * INCHES_TO_M],
                [BLACK_SPOT_X - SQRT_3_INCHES, -1 * INCHES_TO_M],
                [BLACK_SPOT_X, 2 * INCHES_TO_M],
                [BLACK_SPOT_X + SQRT_3_INCHES, INCHES_TO_M],
                [BLACK_SPOT_X + SQRT_3_INCHES, -3 * INCHES_TO_M],
                [BLACK_SPOT_X + 2 * SQRT_3_INCHES, 4 * INCHES_TO_M],
                [BLACK_SPOT_X + 2 * SQRT_3_INCHES, 0],
                [BLACK_SPOT_X + 2 * SQRT_3_INCHES, -2 * INCHES_TO_M],
                [BLACK_SPOT_X, 0],
            ]),
            np.array([3, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2])
        )
        return

    def handle_pocketed_balls(self, dt):
        # Pocket positions
        corner_pocket = np.array([CORNER_POCKET_X, CORNER_POCKET_Y])
        middle_pocket = np.array([MIDDLE_POCKET_X, MIDDLE_POCKET_Y])

        # Distance to the nearest pocket
        dist_corner = np.linalg.norm(np.abs(self.positions) - corner_pocket, axis=1)
        dist_middle = np.linalg.norm(np.abs(self.positions) - middle_pocket, axis=1)
        dist_to_pocket = np.minimum(dist_corner, dist_middle)

        # A ball is in play if it's outside any pocket radius
        self.in_play = (dist_to_pocket >= POCKET_RADIUS) & self.in_play

        # Update positions of in-play balls
        self.positions[self.in_play] += self.velocities[self.in_play] * dt

        # Handle pocketed balls
        not_in_play = ~self.in_play
        self.positions[not_in_play] = np.random.randint(4, 100, size=(not_in_play.sum(), 2))
        self.velocities[not_in_play] = 0.0
        self.angular[not_in_play] = 0.0

    def handle_cushions(self, dt):

        # Cushion Collisions:
        def point_to_segment_dist(points, p0, p1):
            v = np.array(p1) - np.array(p0)
            w = points - np.array(p0)
            t = np.clip((w @ v) / (v @ v), 0, 1)
            proj = p0 + t[:, None] * v
            return np.linalg.norm(points - proj, axis=1)

        def point_to_circle_dist(points, center, radius):
            rel = points - np.array(center)
            d = np.linalg.norm(rel, axis=1)
            return np.abs(d - radius)

        def cushion_candidate_mask(points, radii, segs, circs):
            """
            points: (N,2) array of ball centers
            radii: (N,) array of ball radii
            line_segments: list of (p0, p1) tuples
            circles: list of (cx, cy, r) tuples

            Returns:
                mask: (N,) boolean, True if ball collides
                collision_idx: (N,) int, index of shape (line or circle) giving min distance
                               lines are indexed 0..len(line_segments)-1
                               circles are indexed len(line_segments)..len(line_segments)+len(circles)-1
            """
            N = points.shape[0]
            dists = np.full(N, np.inf)
            coll_idx = np.full(N, -1, dtype=int)  # -1 means no collision

            # check lines
            for idx, (p0, p1) in enumerate(segs):
                d = point_to_segment_dist(points, np.array(p0), np.array(p1))
                m = d < dists
                dists[m] = d[m]
                coll_idx[m] = idx

            # check circles
            offset = len(segs)
            for idx, (cx, cy, r) in enumerate(circs):
                d = point_to_circle_dist(points, np.array([cx, cy]), r)
                m = d < dists
                dists[m] = d[m]
                coll_idx[m] = offset + idx

            m = dists <= radii
            return m, coll_idx

        def cushion_contact_points(points, velocities, radii, segs, circs, coll_idx, delta):
            """
            points: (N,2) ball positions
            velocities: (N,2) ball velocities
            radii: (N,)
            collision_idx: (N,) index of shape hit (from cushion_candidate_mask_with_index)
            dt: timestep

            Returns:
                contact_points: (N,2)
                normals: (N,2)
                tangents: (N,2)
                dt_contact: (N,) time fraction to collision
            """
            contact_points = np.zeros_like(points)
            n = np.zeros_like(points)
            t = np.zeros_like(points)

            line_count = len(segs)

            # Separate masks for lines and circles
            line_mask = coll_idx < line_count
            circle_mask = coll_idx >= line_count

            # --- Lines ---
            if np.any(line_mask):
                idxs = coll_idx[line_mask]
                p0s = np.array([segs[i][0] for i in idxs])
                p1s = np.array([segs[i][1] for i in idxs])
                pts = points[line_mask]

                v = p1s - p0s
                w = pts - p0s
                tau = np.clip(np.sum(w * v, axis=1) / np.sum(v * v, axis=1), 0, 1)
                proj = p0s + (tau[:, None] * v)

                contact_points[line_mask] = proj
                line_dirs = v / np.linalg.norm(v, axis=1)[:, None]
                t[line_mask] = line_dirs
                n[line_mask] = np.column_stack([-line_dirs[:, 1], line_dirs[:, 0]])

            # --- Circles ---
            if np.any(circle_mask):
                idxs = coll_idx[circle_mask] - line_count
                centers = np.array([circs[i][:2] for i in idxs])
                radii_circ = np.array([circs[i][2] for i in idxs])
                pts = points[circle_mask]

                vecs = pts - centers
                norms = np.linalg.norm(vecs, axis=1)[:, None]
                n_vecs = vecs / norms

                contact_points[circle_mask] = centers + n_vecs * radii_circ[:, None]
                n[circle_mask] = n_vecs
                t[circle_mask] = np.column_stack([-n_vecs[:, 1], n_vecs[:, 0]])

            # --- Interpolation along velocity ---
            # Distance from point to cushion at start
            dist_old = np.linalg.norm(points - contact_points, axis=1) - radii
            speed = np.linalg.norm(velocities, axis=1)
            # Avoid division by zero
            alpha = np.where(speed > 0, dist_old / speed / delta, 0)
            alpha = np.clip(alpha, 0, 1)
            dt_c = alpha * delta

            return contact_points, n, t, dt_c

        # --- Check if a ball ends up inside the cushion --- #
        # Broad Check: Filter balls outside (playing_area - radius)
        # Check if ball is in a cushion
        half_w = self.table_width / 2 - self.radii
        half_h = self.table_height / 2 - self.radii
        broad_mask = (np.abs(self.positions[:, 0]) > half_w * 0.98) | \
                     (np.abs(self.positions[:, 1]) > half_h * 0.98)

        candidates = np.where(broad_mask)[0]

        line_segments = [
            ((0.1346, 0.4572), (0.8147, 0.4572)),
            ((0.868785228149406, 0.474264), (0.924562160209359, 0.5162)),
            ((0.0499787766576, 0.5162122073954), (0.078405956002, 0.4826978086535)),
            ((0.1346, -0.4572), (0.8147, -0.4572)),
            ((0.868785228149406, -0.474264), (0.924562160209359, -0.5162)),
            ((0.0499787766576, -0.5162122073954), (0.078405956002, -0.4826978086535)),
            ((-0.1346, 0.4572), (-0.8147, 0.4572)),
            ((-0.868785228149406, 0.474264), (-0.924562160209359, 0.5162)),
            ((-0.0499787766576, 0.5162122073954), (-0.078405956002, 0.4826978086535)),
            ((-0.1346, -0.4572), (-0.8147, -0.4572)),
            ((-0.868785228149406, -0.474264), (-0.924562160209359, -0.5162)),
            ((-0.0499787766576, -0.5162122073954), (-0.078405956002, -0.4826978086535)),
            ((0.9144, -0.3575), (0.9144, 0.3575)),
            ((0.932464, 0.41158522815), (0.9744, 0.467362160211)),
            ((0.932464, -0.41158522815), (0.9744, -0.467362160211)),
            ((-0.9144, -0.3575), (-0.9144, 0.3575)),
            ((-0.932464, 0.41158522815), (-0.9744, 0.467362160211)),
            ((-0.932464, -0.41158522815), (-0.9744, -0.467362160211))
        ]

        circles = [(1.0044, 0.3575, 0.09), (1.0044, -0.3575, 0.09),
                   (-1.0044, 0.3575, 0.09), (-1.0044, -0.3575, 0.09),
                   (0.8147, 0.5472, 0.09), (0.1346, 0.5322, 0.075),
                   (0.8147, -0.5472, 0.09), (0.1346, -0.5322, 0.075),
                   (-0.8147, 0.5472, 0.09), (-0.1346, 0.5322, 0.075),
                   (-0.8147, -0.5472, 0.09), (-0.1346, -0.5322, 0.075)]

        mask, collision_idx = cushion_candidate_mask(
            self.positions[candidates],
            self.radii[candidates],
            line_segments,
            circles
        )

        colliding_balls = candidates[mask]
        contact_pts, normals, tangents, dt_contact = cushion_contact_points(
            self.positions[colliding_balls],
            self.velocities[colliding_balls],
            self.radii[colliding_balls],
            line_segments,
            circles,
            collision_idx[mask],
            dt
        )

        # Move to contact point
        pos_contact = self.positions[colliding_balls] + self.velocities[colliding_balls] * dt_contact[:, None]

        # Reflect velocity along normal
        v_norm = np.sum(self.velocities[colliding_balls] * normals, axis=1)[:, None] * normals
        vel_reflected = self.velocities[colliding_balls] - 2 * v_norm

        # Continue motion for remaining dt
        dt_remain = dt - dt_contact
        pos_final = pos_contact + vel_reflected * dt_remain[:, None]

        # Update balls
        self.positions[colliding_balls] = pos_final
        self.velocities[colliding_balls] = vel_reflected

    def handle_state_updates(self, dt):
        # --- Analytic movement update (from Jia–Mason–Erdmann) ---
        # Slip velocity
        cross_term = self.radii[:, None] * np.column_stack((-self.angular[:, 1], self.angular[:, 0]))
        u0 = self.velocities + cross_term
        s0 = np.linalg.norm(u0, axis=1)
        eps = 1e-12
        uhat = np.zeros_like(u0)
        nonzero = s0 > eps
        uhat[nonzero] = u0[nonzero][:, :2] / s0[nonzero, None]

        # time until slip ends
        t_stop = np.full_like(s0, np.inf, dtype=float)
        if self.mu_s > 0:
            t_stop[nonzero] = (2.0 / 7.0) * s0[nonzero] / (self.mu_s * self.g)

        # classify
        mask_slide = nonzero & (dt <= t_stop)
        mask_trans = nonzero & (dt > t_stop)
        mask_roll = ~nonzero

        # --- Sliding whole step ---
        if np.any(mask_slide):
            idx = np.where(mask_slide)[0]
            self.velocities[idx] -= (self.mu_s * self.g * dt) * uhat[idx]
            ang_factor = (5.0 * self.mu_s * self.g * dt) / (2.0 * self.radii[idx])
            Juhat = np.column_stack((-uhat[idx, 1], uhat[idx, 0]))
            self.angular[idx, :2] += ang_factor[:, None] * Juhat

        # --- Transition from sliding to rolling ---
        if np.any(mask_trans):
            idx = np.where(mask_trans)[0]
            t_rem = dt - t_stop[idx]

            # v1 at slip end (eq. 11)
            v1 = (5.0 / 7.0) * self.velocities[idx] - (2.0 / 7.0) * np.column_stack(
                (-self.radii[idx] * self.angular[idx, 1], self.radii[idx] * self.angular[idx, 0])
            )
            w1_xy = np.column_stack((-v1[:, 1], v1[:, 0])) / self.radii[idx][:, None]

            # roll remainder
            decel = (5.0 / 7.0) * self.mu_r * self.g
            dv_roll = decel * t_rem
            speeds = np.linalg.norm(v1, axis=1)
            vhat = np.zeros_like(v1)
            nz = speeds > eps
            if np.any(nz):
                vhat[nz] = v1[nz] / speeds[nz, None]

            stop = dv_roll >= speeds
            keep = ~stop

            v_final = np.zeros_like(v1)
            w_final = np.zeros_like(w1_xy)

            if np.any(stop):
                v_final[stop] = 0
                w_final[stop] = 0
            if np.any(keep):
                v_final[keep] = v1[keep] - dv_roll[keep, None] * vhat[keep]
                w_final[keep] = np.column_stack((-v_final[keep, 1], v_final[keep, 0])) / self.radii[idx[keep], None]

            self.velocities[idx] = v_final
            self.angular[idx, :2] = w_final

        # --- Already rolling ---
        if np.any(mask_roll):
            idx = np.where(mask_roll)[0]
            speeds = np.linalg.norm(self.velocities[idx], axis=1)
            vhat = np.zeros_like(self.velocities[idx])
            nz = speeds > eps
            if np.any(nz):
                vhat[nz] = self.velocities[idx][nz] / speeds[nz, None]
            decel = (5.0 / 7.0) * self.mu_r * self.g * dt
            stop = decel >= speeds
            keep = ~stop

            if np.any(stop):
                self.velocities[idx[stop]] = 0
                self.angular[idx[stop]] = 0
            if np.any(keep):
                self.velocities[idx[keep]] -= decel * vhat[keep]
                self.angular[idx[keep], :2] = np.column_stack((-self.velocities[idx[keep], 1],
                                                               self.velocities[idx[keep], 0])) / self.radii[
                                                  idx[keep], None]

    def time_step(self):

        v_max = np.max(np.linalg.norm(self.velocities, axis=1))
        w_max = np.max(self.radii * np.linalg.norm(self.angular[:, :2], axis=1))
        speed_scale = max(v_max, w_max)

        dx_target = 0.001
        if speed_scale > 0:
            dt = min(self.dt_max, dx_target / speed_scale)
        else:
            dt = self.dt_max

        self.time += dt

        self.handle_pocketed_balls(dt)
        self.handle_cushions(dt)
        self.handle_state_updates(dt)

        return dt

    def time_step_state_only(self, dt):

        self.time += dt
        # Update positions of in-play balls
        self.positions[self.in_play] += self.velocities[self.in_play] * dt
        self.handle_state_updates(dt)
