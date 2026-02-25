import numpy as np
from pool_simulation.constants import *
from pool_simulation.physics.event import Event
import heapq


class Simulation:
    def __init__(self, n_balls=15,
                 cb_radius=CUE_BALL_RADIUS, cb_mass=CUE_BALL_MASS,
                 ob_radius=OBJECT_BALL_RADIUS, ob_mass=OBJECT_BALL_MASS,
                 mu_s=MU_S, mu_r=MU_R, mu_sp = MU_SP, mu_c = MU_C):

        self.n_balls = n_balls
        self.table_width = TABLE_WIDTH
        self.table_height = TABLE_HEIGHT
        self.cb_mass = cb_mass
        self.cb_radius = cb_radius
        self.ob_mass = ob_mass
        self.ob_radius = ob_radius
        self.mu_s = mu_s  # sliding friction coefficient
        self.mu_r = mu_r  # rolling friction coefficient
        self.mu_sp = mu_sp  # spinning friction coefficient
        self.mu_c = mu_c

        # State arrays
        self.positions = np.zeros((1 + n_balls, 2), dtype=np.float64)
        self.velocities = np.zeros((1 + n_balls, 2), dtype=np.float64)
        self.sliding_velocities = np.zeros((1 + n_balls, 2), dtype=np.float64)
        self.angular = np.zeros((1 + n_balls, 3), dtype=np.float64)
        self.radii = np.array([cb_radius] + [ob_radius] * n_balls, dtype=np.float64)
        self.in_play = np.ones(1 + n_balls, dtype=bool)
        self.colours = np.zeros(1 + n_balls, dtype=np.int8)
        self.ball_states = np.empty(1 + n_balls, dtype="<U10")  # SLIDING | ROLLING | STOPPED | POCKETED
        self.ball_states[:] = "STOPPED"
        self.ball_versions = np.zeros(1 + n_balls, dtype=np.int32)

        self.time = 0.0
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
        self.ball_states[:] = "STOPPED"
        self.time = 0.0
        self.ball_versions.fill(0)

    def reset_to_break(self):
        """Resets the table to break position with micro-gaps for physics stability."""
        try:
            assert self.n_balls == 15
        except AssertionError:
            raise AttributeError("Number of balls must be 15 to reset to break.")

        # Define the perfect mathematical rack
        base_positions = np.array([
            [-0.547, 0],  # Cue ball
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
        ])

        # Generate a microscopic random jitter
        jitter = np.zeros_like(base_positions)
        jitter_magnitude = 5e-5
        jitter[1:] = (np.random.rand(15, 2) - 0.5) * jitter_magnitude

        # Add a tiny bias pushing them slightly away from the apex
        # to guarantee they aren't overlapping
        for i in range(1, 16):
            direction_from_apex = base_positions[i] - base_positions[1]
            if np.linalg.norm(direction_from_apex) > 0:
                jitter[i] += (direction_from_apex / np.linalg.norm(direction_from_apex)) * 1e-5

        final_positions = base_positions + jitter

        self.reset(
            final_positions,
            np.array([3, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2])
        )

    def push_event(self, event):
        heapq.heappush(self.event_queue, event)

    def pop_event(self):
        return heapq.heappop(self.event_queue)

    def get_next_valid_event(self):
        """Pops events until a valid one is found, or the queue is empty."""
        while self.event_queue:
            event = self.pop_event()

            if event.version_i != self.ball_versions[event.i]:
                continue  # Stale event, pop the next one

            if event.j is not None and event.version_j != self.ball_versions[event.j]:
                continue  # Stale event, pop the next one

            return event

        return None  # Queue is empty

    def propel_ball(self, ball_mask, velocities, angulars):
        self.velocities[ball_mask] = velocities
        self.angular[ball_mask] = angulars
        self.ball_states[ball_mask] = "SLIDING"
        self.ball_versions[ball_mask] += 1

    def predict_slide_roll_events(self, ball_mask):
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
        mask = np.zeros(1 + self.n_balls, dtype=bool)
        mask[i] = True
        self.predict_roll_stop_events(mask)
        # self.predict_ball_collision_events(mask)
        # self.predict_cushion_collision_events(mask)
        # self.predict_pot_events(mask)

    def predict_roll_stop_events(self, ball_mask):

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
        mask = np.zeros(1 + self.n_balls, dtype=bool)
        mask[i] = True
        # self.predict_ball_collision_events(mask)
        # self.predict_cushion_collision_events(mask)
        # self.predict_pot_events(mask)

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

        coeffs = np.array([A, B, C, D, E])
        coeffs[np.abs(coeffs) < 1e-12] = 0.0

        roots = np.roots(coeffs)

        valid_times = []
        for root in roots:
            if abs(root.imag) < 1e-6:
                t = root.real
                # Time must be strictly positive.
                # We use > 1e-6 to prevent immediate re-collisions caused by floating point drift
                if t > 1e-6:
                    valid_times.append(t)

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

    def predict_ball_collision_events(self, ball_mask):

        next_event_time = self.event_queue[0].t if self.event_queue else self.time + 5.0
        max_dt = next_event_time - self.time

        pairs_to_check = []
        for i in range(self.n_balls + 1):
            if not self.in_play[i]: continue

            for j in range(i + 1, self.n_balls + 1):
                if not self.in_play[j]: continue

                # If neither ball was just updated, their collision is already predicted
                if not (ball_mask[i] or ball_mask[j]):
                    continue

                # If both are stopped, they can't collide
                if self.ball_states[i] == "STOPPED" and self.ball_states[j] == "STOPPED":
                    continue

                pairs_to_check.append((i, j))

        v_norms = np.linalg.norm(self.velocities, axis=1)

        for i, j in pairs_to_check:
            dist = np.linalg.norm(self.positions[i] - self.positions[j])

            # The absolute maximum distance these two balls could close in the time window
            max_closing_dist = (v_norms[i] + v_norms[j]) * max_dt

            # If they are too far apart to touch, skip
            if dist > (self.radii[i] + self.radii[j] + max_closing_dist):
                continue


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
        restitution = 1.0
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
            if j_t_max > self.mu_c * j_n:
                # Slip occurs throughout the collision
                j_t = self.mu_c * j_n
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

        # Invalidate old future events
        self.ball_versions[i] += 1
        self.ball_versions[j] += 1

        # 9. Predict the new future for these two balls
        mask = np.zeros(1 + self.n_balls, dtype=bool)
        mask[i] = True
        mask[j] = True

        self.predict_slide_roll_events(mask)
        self.predict_ball_collision_events(mask)
    def predict_cushion_collision_events(self):
        return

    def predict_pot_events(self):
        return

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
            self.angular[roll_mask, 0] = new_v[:, 1] / self.radii[roll_mask]
            self.angular[roll_mask, 1] = -new_v[:, 0] / self.radii[roll_mask]

        # z-axis spin
        if np.any(self.in_play):
            w_z = self.angular[self.in_play, 2]
            decay = (5.0 * self.mu_sp * g * dt) / (2.0 * self.radii[self.in_play])
            self.angular[self.in_play, 2] = np.sign(w_z) * np.maximum(0.0, np.abs(w_z) - decay)

    def event_loop(self):
        while event := self.get_next_valid_event():
            dt = event.t - self.time

            if dt > 0:
                self.advance_physics_state(dt)
                self.time = event.t  # Update the master simulation clock

            match event.kind:
                case "SLIDE_ROLL":
                    self.evaluate_slide_roll(event)
                case "ROLL_STOP":
                    self.evaluate_roll_stop(event)
                case "BALL_COLLISION":
                    self.evaluate_ball_collision(event)
                case "CUSHION_COLLISION":
                    # self.evaluate_cushion_collision(event)
                    pass
                case "POT":
                    # self.evaluate_pot(event)
                    pass
                case _:
                    raise ValueError(f"Unknown event kind: {event.kind}")

