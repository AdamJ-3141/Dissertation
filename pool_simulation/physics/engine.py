import numpy as np
import numpy.typing as npt
from pool_simulation.constants import *
from event import Event
from stronge_compliant import resolve_collinear_compliant_frictional_inelastic_collision
import heapq


class Simulation:
    def __init__(self, n_balls=15,
                 cb_radius=CUE_BALL_RADIUS, cb_mass=CUE_BALL_MASS,
                 ob_radius=OBJECT_BALL_RADIUS, ob_mass=OBJECT_BALL_MASS,
                 mu_s=MU_S, mu_r=MU_R, mu_sp = MU_SP, mu_b = MU_B, e_c = E_C,
                 mu_c=MU_C, k_n=K_N, beta_n=BETA_N, beta_t=BETA_T):

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
        self.mu_b = mu_b

        # For cushion collisions (stronge)
        self.e_c = e_c  # Cushion coefficient of restitution
        self.mu_c = mu_c  # Cushion friction coefficient (grippiness of the cloth on the rubber)
        self.k_n = k_n  # Cushion normal spring stiffness
        self.beta_n = beta_n  # Normal mass-matrix coefficient
        self.beta_t = beta_t  # Tangential mass-matrix coefficient (1 + mR^2/I = 1 + 2.5 = 3.5)

        # Table Geometry
        self.line_segments = [
            ((-0.9144, -0.4572), (-0.9144, 0.4572)),
            ((-0.9144, 0.4572), (0.9144, 0.4572)),
            ((0.9144, 0.4572), (0.9144, -0.4572)),
            ((0.9144, -0.4572), (-0.9144, -0.4572))
        ]
        self.circles = []

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

    def propel_ball(self, ball_mask: npt.NDArray[np.bool_], velocities, angulars):
        self.velocities[ball_mask] = velocities
        self.angular[ball_mask] = angulars
        self.ball_states[ball_mask] = "SLIDING"
        self.ball_versions[ball_mask] += 1

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
        mask = np.zeros(1 + self.n_balls, dtype=bool)
        mask[i] = True
        self.predict_roll_stop_events(mask)
        self.predict_ball_collision_events(mask)
        self.predict_cushion_collision_events(mask)
        # self.predict_pot_events(mask)

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
        mask = np.zeros(1 + self.n_balls, dtype=bool)
        mask[i] = True
        self.predict_ball_collision_events(mask)
        self.predict_cushion_collision_events(mask)
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

    def predict_ball_collision_events(self, ball_mask: npt.NDArray[np.bool_]):

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

        # Invalidate old future events
        self.ball_versions[i] += 1
        self.ball_versions[j] += 1

        # 9. Predict the new future for these two balls
        mask = np.zeros(1 + self.n_balls, dtype=bool)
        mask[i] = True
        mask[j] = True

        self.predict_slide_roll_events(mask)
        self.predict_ball_collision_events(mask)
        self.predict_cushion_collision_events(mask)

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

                    roots = np.roots(coeffs)
                    for root in roots:
                        if abs(root.imag) < 1e-6:
                            t = root.real
                            if t > 1e-5:
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

                roots = np.roots(coeffs)
                for root in roots:
                    if abs(root.imag) < 1e-6:
                        t = root.real
                        if t > 1e-5:
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
        mask = np.zeros(1 + self.n_balls, dtype=bool)
        mask[i] = True

        self.predict_slide_roll_events(mask)
        self.predict_ball_collision_events(mask)
        self.predict_cushion_collision_events(mask)

    def predict_pot_events(self, ball_mask: npt.NDArray[np.bool_]):
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

    def run(self, framerate: float = None, frame_callback=None, verbose: bool = False):
        """
        Processes the event queue until all active balls have come to a complete stop.
        Call this directly after propelling the cue ball.
        """

        # picks up any balls just hit with propel_ball()
        if not self.event_queue:
            moving_mask = np.asarray(self.ball_states != "STOPPED", dtype=bool)
            if np.any(moving_mask):
                self.predict_slide_roll_events(moving_mask)
                self.predict_ball_collision_events(moving_mask)
                self.predict_cushion_collision_events(moving_mask) # Uncomment when ready
                # self.predict_pot_events(moving_mask)               # Uncomment when ready

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

            pre_state, pre_v = None, None
            pre_states, pre_v_i, pre_v_j = None, None, None
            i, j = None, None

            if verbose:
                print(f"[{self.time:.4f}] EVENT: {event.kind}")
                if event.j is None:
                    i = event.i
                    print(f"  Ball {i} @ Pos: [{self.positions[i, 0]:.4f}, {self.positions[i, 1]:.4f}]")
                    pre_state = self.ball_states[i]
                    pre_v = self.velocities[i].copy()
                elif isinstance(event.j, int):
                    # Ball-to-Ball collision
                    i, j = event.i, event.j
                    print(f"  Ball {i} @ Pos: [{self.positions[i, 0]:.4f}, {self.positions[i, 1]:.4f}]")
                    print(f"  Ball {j} @ Pos: [{self.positions[j, 0]:.4f}, {self.positions[j, 1]:.4f}]")
                    pre_states = (self.ball_states[i], self.ball_states[j])
                    pre_v_i, pre_v_j = self.velocities[i].copy(), self.velocities[j].copy()
                elif isinstance(event.j, tuple):
                    # Ball-to-Cushion collision
                    i = event.i
                    target_type, target_idx = event.j
                    print(f"  Ball {i} @ Pos: [{self.positions[i, 0]:.4f}, {self.positions[i, 1]:.4f}]")
                    print(f"  Target: {target_type.upper()} {target_idx}")
                    pre_state = self.ball_states[i]
                    pre_v = self.velocities[i].copy()


            # resolve the event
            match event.kind:
                case "SLIDE_ROLL":
                    self.evaluate_slide_roll(event)
                case "ROLL_STOP":
                    self.evaluate_roll_stop(event)
                case "BALL_COLLISION":
                    self.evaluate_ball_collision(event)
                case "CUSHION_COLLISION":
                    self.evaluate_cushion_collision(event)
                    pass
                case "POT":
                    # self.evaluate_pot(event)
                    pass
                case _:
                    raise ValueError(f"Unknown event kind: {event.kind}")

            if verbose:
                if event.j is None or isinstance(event.j, tuple):
                    # Single ball state changed (Slide/Roll/Stop or Cushion Bounce)
                    print(f"    -> State: {pre_state} to {self.ball_states[i]}")
                    print(f"    -> Vel:   [{pre_v[0]:.4f}, {pre_v[1]:.4f}] to [{self.velocities[i, 0]:.4f},"
                          f" {self.velocities[i, 1]:.4f}]")
                elif isinstance(event.j, int):
                    # Two balls state changed
                    print(f"    -> Ball {i} State: {pre_states[0]} to {self.ball_states[i]}")
                    print(f"    -> Ball {j} State: {pre_states[1]} to {self.ball_states[j]}")
                    print(f"    -> Ball {i} Vel:   [{pre_v_i[0]:.4f}, {pre_v_i[1]:.4f}] to "
                          f"[{self.velocities[i, 0]:.4f}, {self.velocities[i, 1]:.4f}]")
                    print(f"    -> Ball {j} Vel:   [{pre_v_j[0]:.4f}, {pre_v_j[1]:.4f}] to "
                          f"[{self.velocities[j, 0]:.4f}, {self.velocities[j, 1]:.4f}]")
                print("-" * 50)

            if framerate and frame_callback:
                frame_callback(self)

            if np.all(self.ball_states[self.in_play] == "STOPPED"):
                self.event_queue.clear()
                break