import numpy as np
from pool_simulation.constants import *


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


def cushion_candidate_mask(points, radii, line_segments, circles):
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
    collision_idx = np.full(N, -1, dtype=int)  # -1 means no collision

    # check lines
    for idx, (p0, p1) in enumerate(line_segments):
        d = point_to_segment_dist(points, np.array(p0), np.array(p1))
        mask = d < dists
        dists[mask] = d[mask]
        collision_idx[mask] = idx

    # check circles
    offset = len(line_segments)
    for idx, (cx, cy, r) in enumerate(circles):
        d = point_to_circle_dist(points, np.array([cx, cy]), r)
        mask = d < dists
        dists[mask] = d[mask]
        collision_idx[mask] = offset + idx

    mask = dists <= radii
    return mask, collision_idx


def cushion_contact_points(points, velocities, radii, line_segments, circles, collision_idx, dt):
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
    N = len(points)
    contact_points = np.zeros_like(points)
    normals = np.zeros_like(points)
    tangents = np.zeros_like(points)
    dt_contact = np.zeros(N)

    line_count = len(line_segments)

    # Separate masks for lines and circles
    line_mask = collision_idx < line_count
    circle_mask = collision_idx >= line_count

    # --- Lines ---
    if np.any(line_mask):
        idxs = collision_idx[line_mask]
        p0s = np.array([line_segments[i][0] for i in idxs])
        p1s = np.array([line_segments[i][1] for i in idxs])
        pts = points[line_mask]

        v = p1s - p0s
        w = pts - p0s
        t = np.clip(np.sum(w * v, axis=1) / np.sum(v*v, axis=1), 0, 1)
        proj = p0s + (t[:, None] * v)

        contact_points[line_mask] = proj
        line_dirs = v / np.linalg.norm(v, axis=1)[:, None]
        tangents[line_mask] = line_dirs
        normals[line_mask] = np.column_stack([-line_dirs[:,1], line_dirs[:,0]])

    # --- Circles ---
    if np.any(circle_mask):
        idxs = collision_idx[circle_mask] - line_count
        centers = np.array([circles[i][:2] for i in idxs])
        radii_circ = np.array([circles[i][2] for i in idxs])
        pts = points[circle_mask]

        vecs = pts - centers
        norms = np.linalg.norm(vecs, axis=1)[:, None]
        n_vecs = vecs / norms

        contact_points[circle_mask] = centers + n_vecs * radii_circ[:, None]
        normals[circle_mask] = n_vecs
        tangents[circle_mask] = np.column_stack([-n_vecs[:,1], n_vecs[:,0]])

    # --- Interpolation along velocity ---
    # Distance from point to cushion at start
    dist_old = np.linalg.norm(points - contact_points, axis=1) - radii
    speed = np.linalg.norm(velocities, axis=1)
    # Avoid division by zero
    alpha = np.where(speed > 0, dist_old / speed / dt, 0)
    alpha = np.clip(alpha, 0, 1)
    dt_contact = alpha * dt

    return contact_points, normals, tangents, dt_contact


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

    def reset(self, positions=None, colours=None):
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

        self.positions += self.velocities * dt

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

        # --- Movement Calculations --- #
        omega_x = self.angular[:, 0]
        omega_y = self.angular[:, 1]
        cross_term = self.radii[:, None] * np.column_stack((-omega_y, omega_x))
        self.sliding_velocities = self.velocities + cross_term
        norms = np.linalg.norm(self.sliding_velocities, axis=1, keepdims=True)
        speeds = np.linalg.norm(self.velocities, axis=1)
        ang_speeds = np.linalg.norm(self.angular, axis=1)
        stopped_mask = np.logical_and(speeds == 0, ang_speeds == 0)
        stopping_mask = ((speeds > 0) & (speeds < 0.001) & (ang_speeds > 0) & (ang_speeds < 0.01))
        sliding_mask = np.logical_and(norms.squeeze() > 0.01, np.logical_not(stopped_mask))
        rolling_mask = np.logical_and(norms.squeeze() <= 0.01, np.logical_not(stopped_mask))

        conditions = [sliding_mask, rolling_mask, stopping_mask, stopped_mask]
        choices = ["Sliding", "Rolling", "Stopping", "Stopped"]
        self.ball_states = np.select(conditions, choices)
        # print(self.ball_states)
        # --- Handle Sliding --- #
        if np.any(sliding_mask):
            s_sliding_velocities = self.sliding_velocities[sliding_mask]
            s_norms = norms[sliding_mask]
            s_radii = self.radii[sliding_mask]
            u_hat = s_sliding_velocities / s_norms
            acc = -self.mu_s * self.g * u_hat
            ang_acc = (5.0 / (2.0 * s_radii[:, None])) * self.mu_s * self.g * np.column_stack(
                (-u_hat[:, 1], u_hat[:, 0], np.zeros(len(s_radii))))
            self.velocities[sliding_mask] += acc * dt
            self.angular[sliding_mask] += ang_acc * dt

        # --- Handle Rolling --- #
        if np.any(rolling_mask):
            r_velocities = self.velocities[rolling_mask]
            r_radii = self.radii[rolling_mask]
            r_speeds = np.linalg.norm(r_velocities, axis=1, keepdims=True)
            r_directions = r_velocities / r_speeds
            deceleration = - (5 / 7) * self.mu_r * self.g
            acc = deceleration * r_directions
            self.velocities[rolling_mask] += acc * dt
            new_velocities = self.velocities[rolling_mask]
            cross_product_v = np.column_stack((-new_velocities[:, 1], new_velocities[:, 0], np.zeros(len(r_radii))))
            self.angular[rolling_mask] = cross_product_v / r_radii[:, None]

        # --- Handle Stopping --- #
        if np.any(stopping_mask):
            self.velocities[stopping_mask] = np.zeros(self.velocities[stopping_mask].shape)
            self.angular[stopping_mask] = np.zeros(self.angular[stopping_mask].shape)

        return dt
