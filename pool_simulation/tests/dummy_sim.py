from pool_simulation.constants import *
import numpy as np


class DummySim:
    def __init__(self, table_width=TABLE_WIDTH, table_height=TABLE_HEIGHT):
        self.n_balls = 0
        self.table_width = table_width
        self.table_height = table_height
        self.cb_mass = CUE_BALL_MASS
        self.cb_radius = CUE_BALL_RADIUS
        self.ob_mass = OBJECT_BALL_MASS
        self.ob_radius = OBJECT_BALL_RADIUS
        self.dt = 0.001

        # State arrays
        self.positions = np.zeros((0, 2), dtype=np.float64)
        self.velocities = np.zeros((0, 2), dtype=np.float64)
        self.spins = np.zeros((0, 2), dtype=np.float64)
        self.angular_velocities = np.zeros(0, dtype=np.float64)
        self.cb_pos = np.zeros((1, 2), dtype=np.float64)
        self.cb_spin = np.zeros((1, 2), dtype=np.float64)
        self.cb_ang_vel = np.zeros(1, dtype=np.float64)
        self.cb_vel = np.zeros((1, 2), dtype=np.float64)

        self.colours = np.zeros(0, dtype=np.int8)
