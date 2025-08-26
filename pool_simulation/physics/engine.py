import numpy as np
from pool_simulation.constants import *


class Simulation:
    def __init__(self, n_balls, table_width, table_height, cb_radius, cb_mass, ob_radius, ob_mass):
        self.n_balls = n_balls
        self.table_width = table_width
        self.table_height = table_height
        self.cb_mass = cb_mass
        self.cb_radius = cb_radius
        self.ob_mass = ob_mass
        self.ob_radius = ob_radius
        self.dt = 0.001

        # State arrays
        self.positions = np.zeros((n_balls, 2), dtype=np.float64)
        self.velocities = np.zeros((n_balls, 2), dtype=np.float64)
        self.angular_velocities = np.zeros((n_balls, 2), dtype=np.float64)
        self.cb_pos = np.zeros((1, 2), dtype=np.float64)
        self.cb_vel = np.zeros((1, 2), dtype=np.float64)
        self.cb_ang_vel = np.zeros((1, 2), dtype=np.float64)

        self.colours = np.zeros(n_balls, dtype=np.int8)

    def reset(self, ob_positions=None, cb_position=None, ob_colours=None):
        """Reset to given positions/velocities, or random if None."""
        if ob_positions is None:
            self.positions[:] = np.random.rand(self.n_balls, 2) * [
                self.table_width, self.table_height
            ]
        else:
            self.positions[:] = ob_positions

        if ob_colours is None:
            self.colours[:] = 0
            cutoff = np.random.randint(0, self.n_balls + 1)
            self.colours[cutoff:self.n_balls] = 1
            self.colours[-1] = 2
        else:
            self.colours[:] = ob_colours

        if cb_position is None:
            self.cb_pos[:] = np.random.randn(1, 2) * [
                self.table_width, self.table_height
            ]
        else:
            self.cb_pos[:] = cb_position

        self.angular_velocities.fill(0.0)
        self.velocities.fill(0.0)
        self.cb_vel.fill(0.0)
        self.cb_ang_vel.fill(0.0)

    def reset_to_break(self):
        """
        Resets the table to break position
        :return:
        """
        self.reset(
            np.array([
                [BLACK_SPOT_X - 2*SQRT_3_INCHES, 0],
                [BLACK_SPOT_X - SQRT_3_INCHES, INCHES_TO_M],
                [BLACK_SPOT_X, -2*INCHES_TO_M],
                [BLACK_SPOT_X + SQRT_3_INCHES, 3*INCHES_TO_M],
                [BLACK_SPOT_X + SQRT_3_INCHES, -1*INCHES_TO_M],
                [BLACK_SPOT_X + 2*SQRT_3_INCHES, 2*INCHES_TO_M],
                [BLACK_SPOT_X + 2*SQRT_3_INCHES, -4*INCHES_TO_M],
                [BLACK_SPOT_X - SQRT_3_INCHES, -1*INCHES_TO_M],
                [BLACK_SPOT_X, 2*INCHES_TO_M],
                [BLACK_SPOT_X + SQRT_3_INCHES, INCHES_TO_M],
                [BLACK_SPOT_X + SQRT_3_INCHES, -3*INCHES_TO_M],
                [BLACK_SPOT_X + 2*SQRT_3_INCHES, 4*INCHES_TO_M],
                [BLACK_SPOT_X + 2*SQRT_3_INCHES, 0],
                [BLACK_SPOT_X + 2*SQRT_3_INCHES, -2*INCHES_TO_M],
                [BLACK_SPOT_X, 0],
            ]),
            np.array([-0.547, 0]),
            np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2])
        )

    def step(self):
        """Update all balls one timestep."""
        self.positions += self.velocities * self.dt
        self.handle_cushions()

    def handle_cushions(self):
        pass
