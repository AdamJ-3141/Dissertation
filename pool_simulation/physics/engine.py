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
