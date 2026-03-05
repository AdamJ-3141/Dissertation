import numpy as np
import numpy.typing as npt
from pool_simulation.constants import *
from .event import Event
from .stronge_compliant import resolve_collinear_compliant_frictional_inelastic_collision
from .solvers import fast_quartic_roots, fast_quadratic_roots
import heapq
import time


class Simulation:
    def __init__(self, n_obj_balls=15,
                 cb_radius=CUE_BALL_RADIUS, cb_mass=CUE_BALL_MASS,
                 ob_radius=OBJECT_BALL_RADIUS, ob_mass=OBJECT_BALL_MASS,
                 mu_s=MU_S, mu_r=MU_R, mu_sp = MU_SP, mu_b = MU_B, e_c = E_C,
                 mu_c=MU_C, k_n=K_N, beta_n=BETA_N, beta_t=BETA_T):

        self.n_obj_balls = n_obj_balls
        self.table_width = TABLE_WIDTH
        self.table_height = TABLE_HEIGHT
        self.cb_mass = cb_mass
        self.cb_radius = cb_radius
        self.ob_mass = ob_mass
        self.ob_radius = ob_radius
        self.mu_s = mu_s  # sliding friction coefficient
        self.mu_r = mu_r  # rolling friction coefficient
        self.mu_sp = mu_sp  # spinning friction coefficient
        self.mu_b = mu_b

        # For cushion collisions (stronge)
        self.e_c = e_c  # Cushion coefficient of restitution
        self.mu_c = mu_c  # Cushion friction coefficient (grippiness of the cloth on the rubber)
        self.k_n = k_n  # Cushion normal spring stiffness
        self.beta_n = beta_n  # Normal mass-matrix coefficient
        self.beta_t = beta_t  # Tangential mass-matrix coefficient (1 + mR^2/I = 1 + 2.5 = 3.5)

        # Table Geometry
        self.line_segments = [
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

        self.circles = [
            (1.0044, 0.3575, 0.09), (1.0044, -0.3575, 0.09),
            (-1.0044, 0.3575, 0.09), (-1.0044, -0.3575, 0.09),
            (0.8147, 0.5472, 0.09), (0.1346, 0.5322, 0.075),
            (0.8147, -0.5472, 0.09), (0.1346, -0.5322, 0.075),
            (-0.8147, 0.5472, 0.09), (-0.1346, 0.5322, 0.075),
            (-0.8147, -0.5472, 0.09), (-0.1346, -0.5322, 0.075)
        ]

        self.pockets = [
            (-CORNER_POCKET_X, CORNER_POCKET_Y, POCKET_RADIUS),  # Top Left
            (0.0, MIDDLE_POCKET_Y, POCKET_RADIUS),  # Top Middle
            (CORNER_POCKET_X, CORNER_POCKET_Y, POCKET_RADIUS),  # Top Right
            (-CORNER_POCKET_X, -CORNER_POCKET_Y, POCKET_RADIUS),  # Bottom Left
            (0.0, -MIDDLE_POCKET_Y, POCKET_RADIUS),  # Bottom Middle
            (CORNER_POCKET_X, -CORNER_POCKET_Y, POCKET_RADIUS)  # Bottom Right
        ]

        # State arrays
        self.positions = np.zeros((1 + n_obj_balls, 2), dtype=np.float64)
        self.velocities = np.zeros((1 + n_obj_balls, 2), dtype=np.float64)
        self.sliding_velocities = np.zeros((1 + n_obj_balls, 2), dtype=np.float64)
        self.angular = np.zeros((1 + n_obj_balls, 3), dtype=np.float64)
        self.radii = np.array([cb_radius] + [ob_radius] * n_obj_balls, dtype=np.float64)
        self.in_play = np.ones(1 + n_obj_balls, dtype=bool)
        self.colours = np.zeros(1 + n_obj_balls, dtype=np.int8)
        self.ball_states = np.empty(1 + n_obj_balls, dtype="<U10")  # SLIDING | ROLLING | STOPPED | POCKETED
        self.ball_states[:] = "STOPPED"
        self.ball_versions = np.zeros(1 + n_obj_balls, dtype=np.int32)

        self.time = 0.0
        self.event_queue = []

        # Numba takes ~0.6s to compile the Stronge model on the first call.
        # We call it here with dummy physics values so the first real shot is instant.
        try:
            from pool_simulation.physics.stronge_compliant import \
                resolve_collinear_compliant_frictional_inelastic_collision

            dummy_beta_t = 3.5
            dummy_beta_n = 1.0
            dummy_eta_sq = (dummy_beta_t / dummy_beta_n) / (1.7 ** 2)

            _ = resolve_collinear_compliant_frictional_inelastic_collision(
                v_t_0=-1.0,  # Must be <= 0
                v_n_0=-1.0,  # Must be < 0
                m=0.17,  # Standard ball mass
                beta_t=dummy_beta_t,
                beta_n=dummy_beta_n,
                mu=0.2,
                e_n=0.85,
                k_n=1e3,
                eta_squared=dummy_eta_sq
            )
        except Exception as e:
            print(f"Warning: Numba warm-up failed. First collision will be slow. ({e})")

        try:
            from pool_simulation.physics.solvers import (
                fast_quadratic_roots, fast_cubic_roots, fast_quartic_roots
            )
            # Pass arbitrary dummy floats just to force Numba to load the cache
            _ = fast_quadratic_roots(1.0, -3.0, 2.0)
            _ = fast_cubic_roots(1.0, -6.0, 11.0, -6.0)
            _ = fast_quartic_roots(1.0, -10.0, 35.0, -50.0, 24.0)
        except Exception as e:
            print(f"Warning: Numba solver warm-up failed. First collision will be slow. ({e})")

    def reset(self, positions=None, colours=None, in_play=None):
        """Reset to given positions, or random if None."""
        if positions is None:
            self.positions[:] = np.random.rand(self.n_obj_balls, 2) * [
                self.table_width, self.table_height
            ]
        else:
            self.positions[:] = positions

        if colours is None:
            self.colours[:] = 0
            cutoff = np.random.randint(0, self.n_obj_balls + 1)
            self.colours[cutoff:self.n_obj_balls] = 1
            self.colours[-1] = 2
        else:
            self.colours[:] = colours

        if in_play is None:
            self.in_play = np.ones_like(self.colours, dtype=bool)
        else:
            self.in_play[:] = in_play

        self.angular.fill(0.0)
        self.velocities.fill(0.0)
        self.ball_states[:] = "STOPPED"
        self.time = 0.0
        self.ball_versions.fill(0)

    def reset_to_break(self):
        """Resets the table to break position with micro-gaps for physics stability."""
        try:
            assert self.n_obj_balls == 15
        except AssertionError:
            raise AttributeError("Number of balls must be 15 to reset to break.")

        R = self.radii[1] + 1e-5
        R_SQRT3 = np.sqrt(3) * R

        base_positions = np.array([
            [-0.547, 0.0],  # Cue ball
            [BLACK_SPOT_X - 2 * R_SQRT3, 0.0],
            [BLACK_SPOT_X - R_SQRT3, R],
            [BLACK_SPOT_X, -2 * R],
            [BLACK_SPOT_X + R_SQRT3, 3 * R],
            [BLACK_SPOT_X + R_SQRT3, -1 * R],
            [BLACK_SPOT_X + 2 * R_SQRT3, 2 * R],
            [BLACK_SPOT_X + 2 * R_SQRT3, -4 * R],
            [BLACK_SPOT_X - R_SQRT3, -1 * R],
            [BLACK_SPOT_X, 2 * R],
            [BLACK_SPOT_X + R_SQRT3, R],
            [BLACK_SPOT_X + R_SQRT3, -3 * R],
            [BLACK_SPOT_X + 2 * R_SQRT3, 4 * R],
            [BLACK_SPOT_X + 2 * R_SQRT3, 0.0],
            [BLACK_SPOT_X + 2 * R_SQRT3, -2 * R],
            [BLACK_SPOT_X, 0.0],
        ])

        # 3. Generate a microscopic random jitter
        jitter = np.zeros_like(base_positions)
        jitter_magnitude = 5e-5
        jitter[1:] = (np.random.rand(15, 2) - 0.5) * jitter_magnitude

        # 4. Add a tiny bias pushing them slightly away from the apex
        for i in range(1, 16):
            direction_from_apex = base_positions[i] - base_positions[1]
            dist = np.linalg.norm(direction_from_apex)
            if dist > 0:
                jitter[i] += (direction_from_apex / dist) * 1e-5

        final_positions = base_positions + jitter

        # 5. Reset the engine with the matched WEPF color array
        self.reset(
            final_positions,
            np.array([0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3])
        )

    def reset_to_six_red(self):
        try:
            assert self.n_obj_balls == 6
        except AssertionError:
            raise AttributeError("Number of balls must be 6 to reset to six-red break.")

        # Define the perfect mathematical rack
        base_positions = np.array([
            [-0.547, 0],  # Cue ball
            [BLACK_SPOT_X - 2 * SQRT_3_INCHES, 0],
            [BLACK_SPOT_X - SQRT_3_INCHES, INCHES_TO_M],
            [BLACK_SPOT_X, -2 * INCHES_TO_M],
            [BLACK_SPOT_X - SQRT_3_INCHES, -1 * INCHES_TO_M],
            [BLACK_SPOT_X, 2 * INCHES_TO_M],
            [BLACK_SPOT_X, 0],
        ])

        # Generate a microscopic random jitter
        jitter = np.zeros_like(base_positions)
        jitter_magnitude = 5e-6
        jitter[1:] = (np.random.rand(6, 2) - 0.5) * jitter_magnitude

        # Add a tiny bias pushing them slightly away from the apex
        # to guarantee they aren't overlapping
        for i in range(1, 7):
            direction_from_apex = base_positions[i] - base_positions[1]
            if np.linalg.norm(direction_from_apex) > 0:
                jitter[i] += (direction_from_apex / np.linalg.norm(direction_from_apex)) * 1e-5

        final_positions = base_positions + jitter

        self.reset(
            final_positions,
            # np.array([0, 1, 1, 1, 1, 1, 1])
            np.array([0, -1, -2, -3, -4, -5, -6])
        )

    def set_up_randomly(self, num_balls: int):
        """Randomly scatters balls, resolving overlaps by resting them microscopically close."""
        if num_balls > self.n_obj_balls:
            raise ValueError(f"Cannot place {num_balls} object balls, max is {self.n_obj_balls}")

        self.positions.fill(0.0)
        self.velocities.fill(0.0)
        self.angular.fill(0.0)
        self.in_play.fill(False)
        self.ball_states.fill("STOPPED")
        self.event_queue = []
        self.time = 0.0
        self.ball_versions.fill(0)

        # 1. Place Cue Ball at the TRUE center
        self.positions[0] = np.array([0.0, 0.0])
        self.in_play[0] = True

        # Table bounds
        x_min, x_max = -0.8, 0.8
        y_min, y_max = -0.4, 0.4

        # Exact physics distance + your microscopic 1e-5 gap
        safe_dist = (self.radii[1] * 2.0) + 1e-5

        # 2. Place Object Balls
        placed = 0
        while placed < num_balls:
            idx = placed + 1
            candidate_pos = np.array([
                np.random.uniform(x_min, x_max),
                np.random.uniform(y_min, y_max)
            ])

            resolved = False
            relaxation_steps = 0

            # Push the ball out of overlaps until it sits perfectly clean
            while not resolved and relaxation_steps < 50:
                active_positions = self.positions[:idx]
                distances = np.linalg.norm(active_positions - candidate_pos, axis=1)

                # Are we clear?
                if np.all(distances >= safe_dist):
                    resolved = True
                    break

                # Find the ball we are overlapping with the most
                worst_idx = np.argmin(distances)

                # Calculate the vector FROM the existing ball TO our candidate
                direction = candidate_pos - active_positions[worst_idx]
                dist_norm = np.linalg.norm(direction)

                # Edge case: If they spawned on the exact same pixel, pick a random direction
                if dist_norm < 1e-8:
                    direction = np.array([np.random.randn(), np.random.randn()])
                    dist_norm = np.linalg.norm(direction)

                direction /= dist_norm

                # Slide the candidate outward along that vector until it microscopically touches
                candidate_pos = active_positions[worst_idx] + (direction * safe_dist)

                # Keep it strictly inside the cushions!
                candidate_pos[0] = np.clip(candidate_pos[0], x_min, x_max)
                candidate_pos[1] = np.clip(candidate_pos[1], y_min, y_max)

                relaxation_steps += 1

            # If it found a safe spot (or successfully clustered), lock it in!
            if resolved:
                self.positions[idx] = candidate_pos
                self.in_play[idx] = True
                self.colours[idx] = 1
                placed += 1
            # Note: If it hits 50 steps without resolving, it means it got trapped in a weird corner.
            # The while loop will safely ignore it and generate a fresh random coordinate for this ball.

        # 3. Initial physics prediction
        mask = self.in_play.copy()
        self.predict_cushion_collision_events(mask)
        self.predict_pot_events(mask)

    def push_event(self, event):
        heapq.heappush(self.event_queue, event)

    def pop_event(self):
        return heapq.heappop(self.event_queue)

    def get_next_valid_event(self):
        """Pops events until a valid one is found, or the queue is empty."""
        while self.event_queue:
            event = self.pop_event()

            # Check if ball i's trajectory has changed
            if event.version_i != self.ball_versions[event.i]:
                continue

            # Check if ball j's trajectory has changed
            if isinstance(event.j, int):
                if event.version_j != self.ball_versions[event.j]:
                    continue

            # If the versions match, the event is valid
            return event

        return None

    def move_cue_ball(self, p: npt.NDArray[np.float64]):
        other_balls_mask = self.in_play.copy()
        other_balls_mask[0] = False

        if np.any(other_balls_mask):
            other_positions = self.positions[other_balls_mask]
            other_radii = self.radii[other_balls_mask]

            dp = other_positions - p
            distances = np.linalg.norm(dp, axis=1)
            safe_distances = self.radii[0] + other_radii + 1e-5

            if np.any(distances < safe_distances):
                raise ValueError("Invalid Placement: Cue ball overlaps with another ball.")

        self.positions[0] = p
        self.ball_versions[0] += 1

        mask = np.zeros(1 + self.n_obj_balls, dtype=bool)
        mask[0] = True
        self.predict_ball_collision_events(mask)
        self.predict_cushion_collision_events(mask)
        self.predict_pot_events(mask)

    def propel_ball(self, ball_mask: npt.NDArray[np.bool_], velocities, angulars):
        self.velocities[ball_mask] = velocities
        self.angular[ball_mask] = angulars
        self.ball_states[ball_mask] = "SLIDING"
        self.ball_versions[ball_mask] += 1

    def strike_cue_ball(self, velocity_x: float, velocity_y: float, topspin: float = 0.0, sidespin: float = 0.0,
                        elevation_deg: float = 0.0):
        """
        topspin: Positive for forward, Negative for backspin (screw/draw).
        sidespin: Positive for right spin (spins CCW), Negative for left english.
        elevation_deg: Cue elevation angle. > 0 tilts the sidespin axis to create swerve.
        """
        v = np.array([velocity_x, velocity_y])
        v_norm = np.linalg.norm(v)

        if v_norm < 1e-8:
            return

        # 1. Base travel vectors
        v_dir = v / v_norm
        v_perp = np.array([-v_dir[1], v_dir[0]])  # Perpendicular points "Left" of the shot line

        # 2. Local Spin Components
        elevation_rad = np.radians(elevation_deg)

        # Spin along the perpendicular axis (Standard Topspin/Draw)
        w_perp = topspin

        # Spin along the vertical Z-axis (Standard Sidespin)
        w_z = sidespin * np.cos(elevation_rad)

        # Spin along the direction of travel (The Massé / Swerve factor!)
        # Elevating the cue tilts the sidespin axis forward into the cloth.
        w_dir = sidespin * np.sin(elevation_rad)

        # 3. Convert local spin to global world coordinates
        w_world_x = (w_dir * v_dir[0]) + (w_perp * v_perp[0])
        w_world_y = (w_dir * v_dir[1]) + (w_perp * v_perp[1])

        angular_velocity = np.array([w_world_x, w_world_y, w_z])

        # 4. Fire the shot!
        active_mask = np.zeros(self.n_obj_balls + 1, dtype=bool)
        active_mask[0] = True

        self.propel_ball(
            ball_mask=active_mask,
            velocities=np.array([v]),
            angulars=np.array([angular_velocity])
        )

    def predict_slide_roll_events(self, ball_mask: npt.NDArray[np.bool_]):
        sliding_mask = self.ball_states == "SLIDING"  # Sliding mask should eliminate any stopped balls

        # u = v + R z-hat x omega
        cross_term = self.radii[:, None] * np.column_stack(
            (-self.angular[:, 1], self.angular[:, 0])
        )
        u = self.velocities + cross_term
        u_norm = np.linalg.norm(u, axis=1)
        moving_mask = u_norm > 1e-6
        final_mask = ball_mask & sliding_mask & moving_mask
        valid_indices = np.where(final_mask)[0]
        for i in valid_indices:
            delta_t = (2.0 / 7.0) * u_norm[i] / (self.mu_s * g)
            event_t = self.time + delta_t
            self.push_event(Event(t=event_t, kind="SLIDE_ROLL", i=i, j=None, version_i=self.ball_versions[i]))
        return

    def evaluate_slide_roll(self, event):
        i = event.i
        self.ball_states[i] = "ROLLING"
        self.ball_versions[i] += 1
        mask = np.zeros(1 + self.n_obj_balls, dtype=bool)
        mask[i] = True
        self.predict_roll_stop_events(mask)
        self.predict_spin_stop_events(mask)
        self.predict_ball_collision_events(mask)
        self.predict_cushion_collision_events(mask)
        self.predict_pot_events(mask)

    def predict_roll_stop_events(self, ball_mask: npt.NDArray[np.bool_]):

        rolling_mask = self.ball_states == "ROLLING"
        v_norm = np.linalg.norm(self.velocities, axis=1)
        moving_mask = v_norm > 1e-6
        final_mask = ball_mask & rolling_mask & moving_mask
        valid_indices = np.where(final_mask)[0]

        # 3. Calculate time to stop and queue events
        for i in valid_indices:
            # Time to stop: t = 7/5 * |v| / (mu_r * g)
            delta_t = (7.0 / 5.0) * v_norm[i] / (self.mu_r * g)
            event_t = self.time + delta_t

            event = Event(t=event_t, kind="ROLL_STOP", i=i, j=None,
                          version_i=self.ball_versions[i])
            self.push_event(event)

    def evaluate_roll_stop(self, event):
        i = event.i
        self.ball_states[i] = "STOPPED"
        self.ball_versions[i] += 1
        mask = np.zeros(1 + self.n_obj_balls, dtype=bool)
        mask[i] = True
        self.predict_ball_collision_events(mask)
        self.predict_spin_stop_events(mask)
        self.predict_cushion_collision_events(mask)
        self.predict_pot_events(mask)

    def predict_spin_stop_events(self, ball_mask):
        """Predicts the exact time a ball's Z-axis spin will decay to zero."""
        spinning_mask = np.abs(self.angular[:, 2]) > 1e-6
        final_mask = ball_mask & spinning_mask & self.in_play
        valid_indices = np.where(final_mask)[0]

        for i in valid_indices:
            w_z = abs(self.angular[i, 2])
            # The exact mathematical decay rate of z-spin
            decay_rate = (5.0 * self.mu_sp * g) / (2.0 * self.radii[i])
            delta_t = w_z / decay_rate
            event_t = self.time + delta_t

            event = Event(
                t=event_t,
                kind="SPIN_STOP",
                i=i,
                j=None,
                version_i=self.ball_versions[i]
            )
            self.push_event(event)

    def evaluate_spin_stop(self, event):
        i = event.i
        # Clamp the spin strictly to 0.0 to prevent floating point drift
        self.angular[i, 2] = 0.0

    def _get_acceleration(self, i):
        state = self.ball_states[i]

        if state == "SLIDING":
            # u = v + R z-hat x omega
            omega_z_cross = np.array([-self.angular[i, 1], self.angular[i, 0]])
            u = self.velocities[i] + self.radii[i] * omega_z_cross
            u_norm = np.linalg.norm(u)
            if u_norm > 1e-8:
                u_hat = u / u_norm
                return -self.mu_s * g * u_hat

        elif state == "ROLLING":
            v = self.velocities[i]
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-8:
                v_hat = v / v_norm
                return -(5.0 / 7.0) * self.mu_r * g * v_hat

        # Stopped balls (or edge cases where norm is about 0) have 0 acceleration
        return np.array([0.0, 0.0])

    def _solve_collision_quartic(self, i, j):
        # Get initial states
        dx = self.positions[i] - self.positions[j]
        dv = self.velocities[i] - self.velocities[j]
        da = self._get_acceleration(i) - self._get_acceleration(j)

        # need the sum of the radii squared for the collision threshold
        R_sum = self.radii[i] + self.radii[j]

        # (1/4 * da.da) t^4 + (da.dv) t^3 + (dv.dv + da.dx) t^2 + (2 * dx.dv) t + (dx.dx - 4R^2) = 0
        A = 0.25 * np.dot(da, da)
        B = np.dot(da, dv)
        C = np.dot(dv, dv) + np.dot(da, dx)
        D = 2.0 * np.dot(dx, dv)
        E = np.dot(dx, dx) - (R_sum ** 2)

        valid_times = []

        # If E is negative, the distance between centers is less than R_sum (they are overlapping).
        # If dot(dx, dv) < 0, they are moving towards each other.
        if E < -1e-5 and np.dot(dx, dv) < 0:
            # Force an immediate collision at t=0.0.
            # This prevents the solver from scheduling the collision for the exit time
            valid_times.append(0.0)
        else:
            roots = fast_quartic_roots(A, B, C, D, E)

            for t in roots:
                if t > -1e-4:
                    valid_times.append(max(0.0, t))

        # Queue the event if a collision happens in the future
        if valid_times:
            # The earliest valid positive root is our collision time
            time_to_impact = min(valid_times)
            absolute_event_time = self.time + time_to_impact

            event = Event(
                t=absolute_event_time,
                kind="BALL_COLLISION",
                i=i,
                j=j,
                version_i=self.ball_versions[i],
                version_j=self.ball_versions[j]
            )
            self.push_event(event)

    def predict_ball_collision_events(self, ball_mask: npt.NDArray[np.bool_]):

        pairs_to_check = []
        for i in range(self.n_obj_balls + 1):
            if not self.in_play[i]: continue

            for j in range(i + 1, self.n_obj_balls + 1):
                if not self.in_play[j]: continue

                # If neither ball was just updated, their collision is already predicted
                if not (ball_mask[i] or ball_mask[j]):
                    continue

                # If both are stopped, they can't collide
                if self.ball_states[i] == "STOPPED" and self.ball_states[j] == "STOPPED":
                    continue

                pairs_to_check.append((i, j))

        for i, j in pairs_to_check:
            self._solve_collision_quartic(i, j)

    def evaluate_ball_collision(self, event):
        i, j = event.i, event.j

        # Gather ball properties
        m1 = self.cb_mass if i == 0 else self.ob_mass
        m2 = self.cb_mass if j == 0 else self.ob_mass
        r1, r2 = self.radii[i], self.radii[j]

        # Moment of inertia for solid spheres: I = 2/5 m r^2
        I1 = (2.0 / 5.0) * m1 * (r1 ** 2)
        I2 = (2.0 / 5.0) * m2 * (r2 ** 2)

        # Calculate the Normal vector (Line of Centers)
        pos_i, pos_j = self.positions[i], self.positions[j]
        delta_pos = pos_j - pos_i
        dist = np.linalg.norm(delta_pos)

        n_hat = delta_pos / dist
        n_hat_3d = np.array([n_hat[0], n_hat[1], 0.0])

        # Calculate Relative Contact Velocity
        # Convert 2D linear velocities to 3D
        v1_3d = np.array([self.velocities[i, 0], self.velocities[i, 1], 0.0])
        v2_3d = np.array([self.velocities[j, 0], self.velocities[j, 1], 0.0])
        w1, w2 = self.angular[i], self.angular[j]

        # Contact point radii vectors
        r1_vec = r1 * n_hat_3d
        r2_vec = -r2 * n_hat_3d

        # Surface velocities at the point of impact
        v_contact1 = v1_3d + np.cross(w1, r1_vec)
        v_contact2 = v2_3d + np.cross(w2, r2_vec)

        # Relative contact velocity (j relative to i)
        v_rel = v_contact2 - v_contact1

        # Normal relative velocity magnitude
        v_rel_n = np.dot(v_rel, n_hat_3d)

        # If they are moving apart, ignore (usually due to micro-collisions/float precision)
        if v_rel_n >= 0:
            return

        # 4. Normal Impulse (Crown assumes perfectly elastic, restitution e = 1)
        # j_n = -(1 + e) * v_rel_n / (1/m1 + 1/m2)
        restitution = RESTITUTION
        j_n = -(1.0 + restitution) * v_rel_n / ((1.0 / m1) + (1.0 / m2))

        # 5. Tangential Impulse (Friction & Spin transfer)
        # Isolate the tangential vector component
        v_rel_t_vec = v_rel - (v_rel_n * n_hat_3d)
        v_rel_t_norm = np.linalg.norm(v_rel_t_vec)

        j_t_vec = np.array([0.0, 0.0, 0.0])

        if v_rel_t_norm > 1e-8:
            t_hat = v_rel_t_vec / v_rel_t_norm

            # The tangential impulse required to completely stop the sliding
            j_t_max = v_rel_t_norm / ((1.0 / m1) + (1.0 / m2) + (r1 ** 2 / I1) + (r2 ** 2 / I2))

            # Crown's threshold: Does it grip, or does it slip?
            if j_t_max > self.mu_b * j_n:
                # Slip occurs throughout the collision
                j_t = self.mu_b * j_n
            else:
                # The surfaces grip/lock together tangentially
                j_t = j_t_max

            # The tangential impulse vector opposes the relative tangential velocity
            j_t_vec = -j_t * t_hat

        # 6. Total Impulse Vector
        J_total = (j_n * n_hat_3d) + j_t_vec

        # 7. Apply Impulses to State
        # Linear velocity (extract x, y)
        self.velocities[i] -= (J_total[:2] / m1)
        self.velocities[j] += (J_total[:2] / m2)

        # Angular velocity
        self.angular[i] -= np.cross(r1_vec, J_total) / I1
        self.angular[j] += np.cross(r2_vec, J_total) / I2

        # 8. Update Engine State
        # Any collision violently alters trajectory, ensuring they enter a sliding phase
        self.ball_states[i] = "SLIDING"
        self.ball_states[j] = "SLIDING"

        # ==========================================
        # POSITIONAL CORRECTION
        # ==========================================
        # Push the balls apart so they are no longer touching.
        # This prevents the solver from getting stuck in an infinite 0.0s collision loop.
        dp = self.positions[i] - self.positions[j]
        dist = np.linalg.norm(dp)

        if dist < 1e-8:
            dp = np.array([1e-8, 0.0])
            dist = 1e-8

        R_sum = self.radii[i] + self.radii[j]

        overlap = R_sum - dist
        if overlap > -1e-5:  # If they are overlapping or perfectly touching
            n_hat = dp / dist
            # Push them apart by the overlap amount + a microscopic 0.01mm gap
            correction = (overlap + 1e-5) / 2.0
            self.positions[i] += n_hat * correction
            self.positions[j] -= n_hat * correction

        # Invalidate old future events
        self.ball_versions[i] += 1
        self.ball_versions[j] += 1

        # 9. Predict the new future for these two balls
        mask = np.zeros(1 + self.n_obj_balls, dtype=bool)
        mask[i] = True
        mask[j] = True

        self.predict_slide_roll_events(mask)
        self.predict_spin_stop_events(mask)
        self.predict_ball_collision_events(mask)
        self.predict_cushion_collision_events(mask)
        self.predict_pot_events(mask)

    def predict_cushion_collision_events(self, ball_mask):
        active_mask = np.asarray(self.in_play & (self.ball_states != "STOPPED") & ball_mask, dtype=bool)
        valid_indices = np.where(active_mask)[0]

        for i in valid_indices:
            P0 = self.positions[i]
            V0 = self.velocities[i]
            A = self._get_acceleration(i)
            R = self.radii[i]

            valid_times = []

            # 1. Check Line Segments (No distance filter!)
            for idx, (p1, p2) in enumerate(self.line_segments):
                P1 = np.array(p1)
                P2 = np.array(p2)

                D = P2 - P1
                L = np.linalg.norm(D)
                if L < 1e-8: continue
                u_hat = D / L
                n_hat = np.array([-u_hat[1], u_hat[0]])

                a_n = np.dot(A, n_hat)
                v_n = np.dot(V0, n_hat)
                p_n = np.dot(P0 - P1, n_hat)

                for offset in [R, -R]:
                    coeffs = np.array([0.5 * a_n, v_n, p_n - offset])
                    coeffs[np.abs(coeffs) < 1e-12] = 0.0

                    roots = fast_quadratic_roots(*coeffs)
                    for t in roots:
                        if t > 1e-6:
                            # Verify the collision happens WITHIN the finite segment bounds
                            P_hit = P0 + (V0 * t) + (0.5 * A * (t ** 2))
                            projection = np.dot(P_hit - P1, u_hat)
                            if 0 <= projection <= L:
                                valid_times.append((t, ('line', idx)))

            # 2. Check Corner Circles (No distance filter!)
            for idx, (cx, cy, cr) in enumerate(self.circles):
                Pc = np.array([cx, cy])
                dp = P0 - Pc
                R_sum = R + cr

                A_c = 0.25 * np.dot(A, A)
                B_c = np.dot(A, V0)
                C_c = np.dot(V0, V0) + np.dot(A, dp)
                D_c = 2.0 * np.dot(dp, V0)
                E_c = np.dot(dp, dp) - (R_sum ** 2)

                coeffs = np.array([A_c, B_c, C_c, D_c, E_c])
                coeffs[np.abs(coeffs) < 1e-12] = 0.0

                roots = fast_quartic_roots(*coeffs)
                for t in roots:
                    if t > 1e-6:
                        valid_times.append((t, ('circle', idx)))

            # 3. Queue the Earliest Event
            if valid_times:
                time_to_impact, target = min(valid_times, key=lambda x: x[0])
                event_t = self.time + time_to_impact

                event = Event(
                    t=event_t,
                    kind="CUSHION_COLLISION",
                    i=i,
                    j=target,
                    version_i=self.ball_versions[i]
                )
                self.push_event(event)

    def evaluate_cushion_collision(self, event):
        """
        Evaluates the resulting state of the balls after collision with a cushion.

        Adapted from:
        https://github.com/ekiefl/pooltool/blob/main/pooltool/physics/resolve/ball_cushion/stronge_compliant/model.py
        :param event: Event
        """
        i = event.i
        target_type, target_idx = event.j

        P_ball = self.positions[i]
        V_ball = self.velocities[i]
        W_ball = self.angular[i]
        R = self.radii[i]
        m = self.cb_mass if i == 0 else self.ob_mass

        # ==========================================
        # 1. Determine the Collision Normal (n_hat)
        # ==========================================

        n_hat = np.empty((1,2), dtype=float)

        if target_type == 'line':
            P1 = np.array(self.line_segments[target_idx][0])
            P2 = np.array(self.line_segments[target_idx][1])
            D = P2 - P1
            u_hat = D / np.linalg.norm(D)
            n_hat = np.array([-u_hat[1], u_hat[0]])

            # Ensure the normal points TOWARDS the ball center from the cushion
            if np.dot(P_ball - P1, n_hat) < 0:
                n_hat = -n_hat

        elif target_type == 'circle':
            Pc = np.array([self.circles[target_idx][0], self.circles[target_idx][1]])
            dp = P_ball - Pc
            n_hat = dp / np.linalg.norm(dp)

        n_3d = np.array([n_hat[0], n_hat[1], 0.0])

        # ==========================================
        # 2. Calculate 3D Contact Surface Velocity
        # ==========================================
        # Convert ball velocity to 3D
        v_3d = np.array([V_ball[0], V_ball[1], 0.0])

        # Vector from ball center to contact point on the cushion
        r_contact = -R * n_3d

        # True velocity of the ball's surface at the point of impact
        v_contact = v_3d + np.cross(W_ball, r_contact)

        # ==========================================
        # 3. Decompose into Normal and Tangent Vectors
        # ==========================================
        # Normal component
        v_n_0 = np.dot(v_contact, n_3d)

        # If moving away from the cushion (e.g., v_n_0 >= 0), it's a micro-collision ghost; ignore it.
        if v_n_0 >= -1e-8:
            return

        # Tangential component vector
        v_t_vec = v_contact - (v_n_0 * n_3d)
        v_t_norm = np.linalg.norm(v_t_vec)

        # Stronge's model strictly expects the tangential direction vector (t_3d) to be
        # oriented such that the initial tangential velocity (v_t_0) is negative.
        if v_t_norm > 1e-8:
            t_3d = -v_t_vec / v_t_norm
            v_t_0 = -v_t_norm
        else:
            t_3d = np.array([0.0, 0.0, 0.0])
            v_t_0 = 0.0

        # ==========================================
        # 4. Apply Stronge's Compliant Impact Model
        # ==========================================
        eta_squared = (self.beta_t / self.beta_n) / (1.7 ** 2)

        v_t_f, v_n_f = resolve_collinear_compliant_frictional_inelastic_collision(
            v_t_0=v_t_0,
            v_n_0=v_n_0,
            m=m,
            beta_t=self.beta_t,
            beta_n=self.beta_n,
            mu=self.mu_c,
            e_n=self.e_c,
            k_n=self.k_n,
            eta_squared=eta_squared
        )

        # ==========================================
        # 5. Apply the Velocity Deltas
        # ==========================================
        # Calculate the change in velocity according to the mass-matrix coefficients
        Dv_n = (v_n_f - v_n_0) / self.beta_n
        Dv_t = (v_t_f - v_t_0) / self.beta_t

        # Update Linear Velocity (Extract x, y)
        delta_v_linear = (Dv_n * n_3d) + (Dv_t * t_3d)
        self.velocities[i] += delta_v_linear[:2]

        # Update Angular Velocity (Transferring friction to spin)
        # delta_w = I^-1 * (r x J) = (2.5 / R) * (-n_hat x Dv_t * t_hat)
        delta_w = (2.5 / R) * np.cross(-n_3d, Dv_t * t_3d)
        self.angular[i] += delta_w

        # ==========================================
        # 6. Update Engine State
        # ==========================================
        self.ball_states[i] = "SLIDING"
        self.ball_versions[i] += 1

        # Predict the new future for this ball
        mask = np.zeros(1 + self.n_obj_balls, dtype=bool)
        mask[i] = True

        self.predict_slide_roll_events(mask)
        self.predict_spin_stop_events(mask)
        self.predict_ball_collision_events(mask)
        self.predict_cushion_collision_events(mask)
        self.predict_pot_events(mask)

    def predict_pot_events(self, ball_mask):
        active_mask = np.asarray(self.in_play & (self.ball_states != "STOPPED") & ball_mask, dtype=bool)
        valid_indices = np.where(active_mask)[0]

        for i in valid_indices:
            P0 = self.positions[i]
            V0 = self.velocities[i]
            A = self._get_acceleration(i)
            R = self.radii[i]

            # Time required for gravity to pull the ball down below the pocket lip
            t_drop = np.sqrt((2.0 * R) / g)

            valid_times = []

            for idx, (cx, cy, pr) in enumerate(self.pockets):
                Pc = np.array([cx, cy])
                dp = P0 - Pc

                if np.dot(dp, dp) < (pr ** 2):
                    valid_times.append((0.0, idx))
                    continue  # Skip the complex chord math for this pocket

                A_c = 0.25 * np.dot(A, A)
                B_c = np.dot(A, V0)
                C_c = np.dot(V0, V0) + np.dot(A, dp)
                D_c = 2.0 * np.dot(dp, V0)
                E_c = np.dot(dp, dp) - (pr ** 2)

                coeffs = np.array([A_c, B_c, C_c, D_c, E_c])
                coeffs[np.abs(coeffs) < 1e-12] = 0.0

                roots = fast_quartic_roots(*coeffs)
                pocket_roots = []

                for t in roots:
                    if t > 1e-5:
                        pocket_roots.append(t)

                # We need both an entry and exit time to evaluate the chord
                if len(pocket_roots) >= 2:
                    pocket_roots.sort()
                    t_entry = pocket_roots[0]
                    t_exit = pocket_roots[1]
                    delta_t = t_exit - t_entry

                    # Find the closest point the ball gets to the center of the pocket
                    t_closest = (t_entry + t_exit) / 2.0
                    P_closest = P0 + (V0 * t_closest) + (0.5 * A * (t_closest ** 2))

                    # Impact Parameter: Distance from pocket center to the trajectory line
                    b_dist = np.linalg.norm(P_closest - Pc)
                    hit_fraction = float(b_dist) / pr

                    # ==========================================
                    # THE CHORD / RATTLE HEURISTIC
                    # ==========================================
                    if hit_fraction < 0.7:
                        # Solid Hit (Center cut or slightly off-center)
                        # The ball hits the back liner of the pocket. It drops regardless of speed.
                        valid_times.append((t_entry, idx))
                    elif delta_t > t_drop:
                        # Grazing Hit (Small chord length) BUT moving slow enough to fall
                        valid_times.append((t_entry, idx))

                    # IF Grazing Hit AND too fast:
                    # We do nothing! The event is ignored, and the cushion predictor
                    # will naturally crash the ball into the far circular jaw instead!

            if valid_times:
                time_to_drop, pocket_idx = min(valid_times, key=lambda x: x[0])
                event_t = self.time + time_to_drop

                event = Event(
                    t=event_t,
                    kind="POT",
                    i=i,
                    j=('pocket', pocket_idx),
                    version_i=self.ball_versions[i]
                )
                self.push_event(event)

    def evaluate_pot(self, event):
        i = event.i

        # Remove from play
        self.in_play[i] = False
        self.ball_states[i] = "STOPPED"

        # Kill all momentum
        self.velocities[i] = np.array([0.0, 0.0])
        self.angular[i] = np.array([0.0, 0.0, 0.0])

        self.positions[i] = np.array([999.0, 999.0])

        self.ball_versions[i] += 1

        # If the cue ball (0) is potted
        if i == 0:
            print("SCRATCH! Cue ball potted.")

    def advance_physics_state(self, dt):
        if dt <= 0.0:
            return

        slide_mask = self.ball_states == "SLIDING"
        if np.any(slide_mask):
            # Contact velocity: u = v + R z-hat x omega
            # z-hat x omega for a 3D omega (wx, wy, wz) crossed with (0,0,1) is (-wy, wx, 0)
            cross_term = self.radii[slide_mask, None] * np.column_stack(
                (-self.angular[slide_mask, 1], self.angular[slide_mask, 0])
            )
            u = self.velocities[slide_mask] + cross_term

            # Sliding direction unit vector: u_hat = u / ||u||
            u_norm = np.linalg.norm(u, axis=1, keepdims=True)
            safe_u_norm = np.where(u_norm > 1e-8, u_norm, 1.0)
            u_hat = u / safe_u_norm

            # Position: x = x0 + v0 t - 1/2 mu_s g t^2 u_hat
            self.positions[slide_mask] += (self.velocities[slide_mask] * dt) - (0.5 * self.mu_s * g * (dt ** 2) * u_hat)

            # Velocity: v = v0 - mu_s g t u_hat
            self.velocities[slide_mask] -= self.mu_s * g * dt * u_hat

            # Angular velocity: omega = omega0 + (5 / 2R) mu_s g t z-hat x u_hat
            # z-hat x u_hat = (-u_hat_y, u_hat_x)
            z_cross_u = np.column_stack((-u_hat[:, 1], u_hat[:, 0]))
            coef = (5.0 * self.mu_s * g * dt) / (2.0 * self.radii[slide_mask, None])
            self.angular[slide_mask, 0:2] += coef * z_cross_u

        roll_mask = self.ball_states == "ROLLING"
        if np.any(roll_mask):
            v1 = self.velocities[roll_mask]
            v_norm = np.linalg.norm(v1, axis=1, keepdims=True)
            safe_v_norm = np.where(v_norm > 1e-8, v_norm, 1.0)
            v1_hat = v1 / safe_v_norm

            # Position: x = x1 + (t-t1) v1 - 5/14 mu_r g (t-t1)^2 v1_hat
            self.positions[roll_mask] += (v1 * dt) - ((5.0 / 14.0) * self.mu_r * g * (dt ** 2) * v1_hat)

            # Velocity: v = v1 - 5/7 mu_r g (t-t1) v1_hat
            self.velocities[roll_mask] -= (5.0 / 7.0) * self.mu_r * g * dt * v1_hat

            # Angular velocity: omega = 1/R z-hat x v
            new_v = self.velocities[roll_mask]
            self.angular[roll_mask, 0] = -new_v[:, 1] / self.radii[roll_mask]
            self.angular[roll_mask, 1] = new_v[:, 0] / self.radii[roll_mask]

        # z-axis spin
        if np.any(self.in_play):
            w_z = self.angular[self.in_play, 2]
            decay = (5.0 * self.mu_sp * g * dt) / (2.0 * self.radii[self.in_play])
            self.angular[self.in_play, 2] = np.sign(w_z) * np.maximum(0.0, np.abs(w_z) - decay)

    def run(self, framerate: float = None, frame_callback=None, verbose: bool = False):
        """
        Processes the event queue until all active balls have come to a complete stop.
        Call this directly after propelling the cue ball.
        """
        start_time = time.perf_counter()
        # picks up any balls just hit with propel_ball()
        if not self.event_queue:
            moving_mask = np.asarray(self.ball_states != "STOPPED", dtype=bool)
            if np.any(moving_mask):
                self.predict_slide_roll_events(moving_mask)
                self.predict_spin_stop_events(moving_mask)
                self.predict_ball_collision_events(moving_mask)
                self.predict_cushion_collision_events(moving_mask)
                self.predict_pot_events(moving_mask)

        if framerate and frame_callback:
            frame_dt = 1.0 / framerate
            next_frame_time = self.time + frame_dt
            frame_callback(self)  # Record the initial starting state
        else:
            next_frame_time = float('inf')
            frame_dt = 0

        # Main Event Loop
        while True:
            # Pop the next valid event
            event = self.get_next_valid_event()

            if event is None:
                if verbose:
                    print(f"\n[{self.time:.4f}] Simulation at rest. Queue empty.")
                break  # The queue is empty - simulation is at rest.

            while next_frame_time < event.t:
                dt = next_frame_time - self.time
                self.advance_physics_state(dt)
                self.time = next_frame_time
                frame_callback(self)
                next_frame_time += frame_dt

            # Advance the global clock and physical state to the exact moment of the event
            dt = event.t - self.time
            if dt > 0:
                self.advance_physics_state(dt)
                self.time = event.t

            pre_state, pre_v, pre_w = None, None, None
            pre_states, pre_v_i, pre_v_j = None, None, None
            i, j = None, None
            pre_w_i, pre_w_j = None, None

            if verbose:
                print(f"[{self.time:.4f}] EVENT: {event.kind}")
                if event.j is None:
                    i = event.i
                    print(f"  Ball {i} @ Pos: [{self.positions[i, 0]:.4f}, {self.positions[i, 1]:.4f}]")
                    pre_state = self.ball_states[i]
                    pre_v = self.velocities[i].copy()
                    pre_w = self.angular[i].copy()
                elif isinstance(event.j, int):
                    # Ball-to-Ball collision
                    i, j = event.i, event.j
                    print(f"  Ball {i} @ Pos: [{self.positions[i, 0]:.4f}, {self.positions[i, 1]:.4f}]")
                    print(f"  Ball {j} @ Pos: [{self.positions[j, 0]:.4f}, {self.positions[j, 1]:.4f}]")
                    pre_states = (self.ball_states[i], self.ball_states[j])
                    pre_v_i, pre_v_j = self.velocities[i].copy(), self.velocities[j].copy()
                    pre_w_i, pre_w_j = self.angular[i].copy(), self.angular[j].copy()
                elif isinstance(event.j, tuple):
                    # Ball-to-Cushion collision
                    i = event.i
                    target_type, target_idx = event.j
                    print(f"  Ball {i} @ Pos: [{self.positions[i, 0]:.4f}, {self.positions[i, 1]:.4f}]")
                    print(f"  Target: {target_type.upper()} {target_idx}")
                    pre_state = self.ball_states[i]
                    pre_v = self.velocities[i].copy()
                    pre_w = self.angular[i].copy()

            start = time.time()
            # resolve the event
            match event.kind:
                case "SLIDE_ROLL":
                    self.evaluate_slide_roll(event)
                case "ROLL_STOP":
                    self.evaluate_roll_stop(event)
                case "SPIN_STOP":
                    self.evaluate_spin_stop(event)
                case "BALL_COLLISION":
                    self.evaluate_ball_collision(event)
                case "CUSHION_COLLISION":
                    self.evaluate_cushion_collision(event)
                    pass
                case "POT":
                    self.evaluate_pot(event)
                    pass
                case _:
                    raise ValueError(f"Unknown event kind: {event.kind}")

            if verbose:
                if event.j is None or isinstance(event.j, tuple):
                    # Single ball state changed (Slide/Roll/Stop or Cushion Bounce)
                    print(f"    -> State: {pre_state} to {self.ball_states[i]}")
                    print(f"    -> Vel:   [{pre_v[0]:.4f}, {pre_v[1]:.4f}] to [{self.velocities[i, 0]:.4f},"
                          f" {self.velocities[i, 1]:.4f}]")
                    print(f"    -> Spin:  [{pre_w[0]:.2f}, {pre_w[1]:.2f}, {pre_w[2]:.2f}] to "
                          f"[{self.angular[i, 0]:.2f}, {self.angular[i, 1]:.2f}, {self.angular[i, 2]:.2f}]")
                elif isinstance(event.j, int):
                    # Two balls state changed
                    print(f"    -> Ball {i} State: {pre_states[0]} to {self.ball_states[i]}")
                    print(f"    -> Ball {j} State: {pre_states[1]} to {self.ball_states[j]}")
                    print(f"    -> Ball {i} Vel:   [{pre_v_i[0]:.4f}, {pre_v_i[1]:.4f}] to "
                          f"[{self.velocities[i, 0]:.4f}, {self.velocities[i, 1]:.4f}]")
                    print(f"    -> Ball {j} Vel:   [{pre_v_j[0]:.4f}, {pre_v_j[1]:.4f}] to "
                          f"[{self.velocities[j, 0]:.4f}, {self.velocities[j, 1]:.4f}]")
                    print(f"    -> Ball {i} Spin:  [{pre_w_i[0]:.2f}, {pre_w_i[1]:.2f}, {pre_w_i[2]:.2f}] "
                          f"to [{self.angular[i, 0]:.2f}, {self.angular[i, 1]:.2f}, {self.angular[i, 2]:.2f}]")
                    print(f"    -> Ball {j} Spin:  [{pre_w_j[0]:.2f}, {pre_w_j[1]:.2f}, {pre_w_j[2]:.2f}] "
                          f"to [{self.angular[j, 0]:.2f}, {self.angular[j, 1]:.2f}, {self.angular[j, 2]:.2f}]")
                print(f"Evaluated in {time.time()-start}")
                print("-" * 50)

            # if framerate and frame_callback:
            #     frame_callback(self)

            is_stopped = np.all(self.ball_states[self.in_play] == "STOPPED")
            is_not_spinning = np.all(np.abs(self.angular[self.in_play, 2]) < 1e-6)

            if is_stopped and is_not_spinning:
                if framerate and frame_callback:
                    frame_callback(self)
                self.event_queue.clear()
                break

        return time.perf_counter() - start_time