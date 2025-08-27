import numpy as np
from pool_simulation.constants import *


class Simulation:
    def __init__(self, n_balls, table_width, table_height, cb_radius, cb_mass, ob_radius, ob_mass, mu_s, mu_r):
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
        self.angular_velocities = np.zeros((1 + n_balls, 3), dtype=np.float64)
        self.radii = np.array([cb_radius] + [ob_radius] * n_balls, dtype=np.float64)
        self.in_play = np.ones(1 + n_balls, dtype=bool)
        self.colours = np.zeros(1 + n_balls, dtype=np.int8)

        # internal simulation clock
        self.time = 0.0

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

        self.angular_velocities.fill(0.0)
        self.velocities.fill(0.0)
        self.phase = ["sliding"] * (1 + self.n_balls)
        self.time = 0.0
        self.schedule_initial_events()

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

    # ---------------------------------------------------
    # Event Scheduling
    # ---------------------------------------------------
    def schedule_initial_events(self):
        """Compute when sliding ends for each ball."""
        z = np.array([0, 0, 1])
        for i in range(len(self.positions)):
            v0 = self.velocities[i]
            w0 = self.angular_velocities[i]
            R = self.radii[i]

            u0 = v0 + R * np.cross(z, w0)[:2]
            if np.linalg.norm(u0) < 1e-9:
                self.phase[i] = "rolling"
                self.next_event_time[i] = np.inf
            else:
                t1 = (2 / 7) * (np.linalg.norm(u0) / (self.mu_s * self.g))
                self.next_event_time[i] = self.time + t1

    # ---------------------------------------------------
    # Advance to Next Event
    # ---------------------------------------------------
    def step_to_next_event(self):
        """Advance simulation directly to the next global event time."""
        # pick the earliest event among all balls
        t_next = np.min(self.next_event_time)
        if np.isinf(t_next):
            return False  # nothing left to simulate

        dt = t_next - self.time
        self._advance_all(dt)
        self.time = t_next

        # process events for balls reaching their event
        for i in range(len(self.positions)):
            if abs(self.next_event_time[i] - t_next) < 1e-9:
                self._handle_event(i)

        return True

    # ---------------------------------------------------
    # Advance positions analytically
    # ---------------------------------------------------
    def _advance_all(self, dt):
        z = np.array([0, 0, 1])
        for i in range(len(self.positions)):
            if self.phase[i] == "sliding":
                v0 = self.velocities[i]
                R = self.radii[i]
                w0 = self.angular_velocities[i]

                u0 = v0 + R * np.cross(z, w0)[:2]
                u_hat = u0 / np.linalg.norm(u0)
                x_new = self.positions[i] + v0 * dt - 0.5 * self.mu_s * self.g * dt ** 2 * u_hat
                v_new = v0 - self.mu_s * self.g * dt * u_hat

                self.positions[i] = x_new
                self.velocities[i] = v_new

            elif self.phase[i] == "rolling":
                v1 = self.velocities[i]
                v1_norm = np.linalg.norm(v1)
                if v1_norm < 1e-9:
                    continue
                v1_hat = v1 / v1_norm
                decel = (5 / 7) * self.mu_r * self.g
                x_new = self.positions[i] + v1 * dt - 0.5 * decel * dt ** 2 * v1_hat
                v_new = v1 - decel * dt * v1_hat

                self.positions[i] = x_new
                self.velocities[i] = v_new

    # ---------------------------------------------------
    # Event Handler
    # ---------------------------------------------------
    def _handle_event(self, i):
        if self.phase[i] == "sliding":
            # transition to rolling
            self.phase[i] = "rolling"
            v1 = self.velocities[i]
            v1_norm = np.linalg.norm(v1)
            if v1_norm < 1e-9:
                self.phase[i] = "stopped"
                self.next_event_time[i] = np.inf
                return
            t2 = self.time + (7 / 5) * (v1_norm / (self.mu_r * self.g))
            self.next_event_time[i] = t2
        elif self.phase[i] == "rolling":
            self.phase[i] = "stopped"
            self.next_event_time[i] = np.inf

