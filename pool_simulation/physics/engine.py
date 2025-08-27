import numpy as np
from pool_simulation.constants import *


class Simulation:
    def __init__(self, n_balls=15, table_width=TABLE_WIDTH, table_height=TABLE_HEIGHT,
                 cb_radius=CUE_BALL_RADIUS, cb_mass=CUE_BALL_MASS,
                 ob_radius=OBJECT_BALL_RADIUS, ob_mass=OBJECT_BALL_MASS,
                 mu_s=MU_S, mu_r=MU_R, dt_max=1.0/60):

        self.n_balls = n_balls
        self.table_width = table_width
        self.table_height = table_height
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

        p_prev = self.positions.copy()
        self.positions += self.velocities * dt

        # --- Check if a ball ends up inside the cushion --- #
        # Filter balls outside (playing_area - radius)
        # Check if ball is in a pocket jaw

        # --- Movement Calculations --- #
        omega_x = self.angular[:, 0]
        omega_y = self.angular[:, 1]
        cross_term = self.radii[:, None] * np.column_stack((-omega_y, omega_x))
        self.sliding_velocities = self.velocities + cross_term
        norms = np.linalg.norm(self.sliding_velocities, axis=1, keepdims=True)
        print(norms)
        speeds = np.linalg.norm(self.velocities, axis=1)
        ang_speeds = np.linalg.norm(self.angular, axis=1)
        stopped_mask = np.logical_and(speeds == 0, ang_speeds == 0)
        stopping_mask = ((speeds > 0) & (speeds < 0.001) & (ang_speeds > 0) & (ang_speeds < 0.01))
        sliding_mask = np.logical_and(norms.squeeze() > 0.01, np.logical_not(stopped_mask))
        rolling_mask = np.logical_and(norms.squeeze() <= 0.01, np.logical_not(stopped_mask))

        conditions = [sliding_mask, rolling_mask, stopping_mask, stopped_mask]
        choices = ["Sliding", "Rolling", "Stopping", "Stopped"]
        self.ball_states = np.select(conditions, choices)
        print(self.ball_states)
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
