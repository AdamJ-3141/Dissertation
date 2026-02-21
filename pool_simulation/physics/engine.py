import numpy as np
from pool_simulation.constants import *
from pool_simulation.physics.event import Event
import heapq


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

        # State arrays
        self.positions = np.zeros((1 + n_balls, 2), dtype=np.float64)
        self.velocities = np.zeros((1 + n_balls, 2), dtype=np.float64)
        self.sliding_velocities = np.zeros((1 + n_balls, 2), dtype=np.float64)
        self.angular = np.zeros((1 + n_balls, 3), dtype=np.float64)
        self.radii = np.array([cb_radius] + [ob_radius] * n_balls, dtype=np.float64)
        self.in_play = np.ones(1 + n_balls, dtype=bool)
        self.colours = np.zeros(1 + n_balls, dtype=np.int8)
        self.ball_states = np.zeros(1 + n_balls, dtype=str)  # SLIDING | ROLLING | STOPPED | POCKETED

        self.event_queue = []

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

    def push_event(self, event):
        heapq.heappush(self.event_queue, event)

    def pop_event(self):
        return heapq.heappop(self.event_queue)

    def predict_slide_roll_events(self):
        # u = v + R z-hat x omega
        cross_term = self.radii[:, None] * np.column_stack(
            (-self.angular[:, 1], self.angular[:, 0])
        )
        u = self.velocities + cross_term
        t1 = 2/7 * np.linalg.norm(u, axis=1, keepdims=True)/(MU_S*g)
        self.push_event(Event())
        return

    def predict_roll_stop_events(self):
        return

    def predict_ball_collision_events(self):
        return

    def predict_cushion_collision_events(self):
        return

    def predict_pot_events(self):
        return
