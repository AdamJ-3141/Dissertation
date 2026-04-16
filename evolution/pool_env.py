import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

from pool_simulation.physics import Simulation


class PoolEnv(gym.Env):
    """Custom PyTorch Environment for the Pool Simulation"""

    # Gymnasium requires metadata for rendering
    metadata = {"render_modes": ["human", "none"], "render_fps": 60}
    REWARD_FIRST_HIT_TARGET = 10
    REWARD_POTTED = 100

    def __init__(self, render_mode="none"):
        super(PoolEnv, self).__init__()

        self.render_mode = render_mode
        self.sim = Simulation(start_break=False)
        self.target_color = 1  # We will start with a 1-ball drill for Phase 1

        # ==========================================
        # ACTION SPACE
        # ==========================================
        # 7 outputs: [vx, vy, topspin, sidespin, elevation, place_x, place_y]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        # ==========================================
        # OBSERVATION SPACE
        # ==========================================
        # [cb_x, cb_y, 15 x rel_x, 15 x rel_y, 15 x target_flags, 4 x turnstate]
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(51,), dtype=np.float32)

        self.latest_shot_data = {}

        self.max_spawn_radius = 0.15
        self.success_history = deque(maxlen=50)
        self.promotion_threshold = 0.80

    def _get_obs(self):
        """Helper method to compile the 51-input radar array for PyTorch."""
        # 1. Cue Ball Absolute Position (Normalized between -1 and 1)
        cb_x = np.array([self.sim.positions[0, 0] / (self.sim.table_width / 2.0)], dtype=np.float32)
        cb_y = np.array([self.sim.positions[0, 1] / (self.sim.table_height / 2.0)], dtype=np.float32)

        # 2. Set up the empty arrays for 15 object balls
        rel_x = np.zeros(15, dtype=np.float32)
        rel_y = np.zeros(15, dtype=np.float32)
        target_flags = np.zeros(15, dtype=np.float32)

        # 3. Fill in data for active target balls
        for i in range(1, len(self.sim.colours)):
            if not self.sim.in_play[i]:
                continue

            is_target = (self.target_color is not None and self.sim.colours[i] == self.target_color)

            if is_target:
                rel_x[i - 1] = (self.sim.positions[i, 0] - self.sim.positions[0, 0]) / self.sim.table_width
                rel_y[i - 1] = (self.sim.positions[i, 1] - self.sim.positions[0, 1]) / self.sim.table_height
                target_flags[i - 1] = 1.0

        # 4. Turn State (One-Hot Encoded). Hardcoded to NORMAL (0) for Phase 1.
        # [NORMAL, BALL_IN_HAND, BALL_IN_HAND_BAULK, GAME_OVER]
        turn_state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # 5. Combine everything into a single 51-element array
        obs = np.concatenate([cb_x, cb_y, rel_x, rel_y, target_flags, turn_state])
        return obs

    def _is_valid_spawn(self, rx, ry, cue_ball_pos):
        """Mathematically verifies a coordinate is on the playable cloth."""

        # 1. Reject if it overlaps the cue ball
        if np.hypot(rx - cue_ball_pos[0], ry - cue_ball_pos[1]) < 0.06:
            return False

        # 2. Reject if it spawns instantly inside the hole (already falling)
        for pocket in self.sim.pockets:
            if np.hypot(rx - pocket[0], ry - pocket[1]) < 0.05:
                return False

        ball_radius = 0.0254  # OBJECT_BALL_RADIUS
        margin = 0.002  # 2mm safety gap

        # 3. Reject if embedded in straight cushions or flat jaws (Point-to-Line distance)
        for p1, p2 in self.sim.line_segments:
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            l2 = dx * dx + dy * dy
            t = max(0.0, min(1.0, ((rx - p1[0]) * dx + (ry - p1[1]) * dy) / (l2 + 1e-8)))

            proj_x = p1[0] + t * dx
            proj_y = p1[1] + t * dy

            if np.hypot(rx - proj_x, ry - proj_y) < (ball_radius + margin):
                return False

        # 4. Reject if embedded in rounded cushion knuckles
        for cx, cy, cr in self.sim.circles:
            if np.hypot(rx - cx, ry - cy) < (cr + ball_radius + margin):
                return False

        # 5. Strict Outer Bounds Check (Fixing the void problem)
        main_bed_x = 1.8288 / 2.0  # TABLE_WIDTH / 2
        main_bed_y = 0.9144 / 2.0  # TABLE_HEIGHT / 2

        if abs(rx) > main_bed_x or abs(ry) > main_bed_y:
            in_jaw = False

            # Middle Pockets Jaw Zone (Cushions break between x = -0.1346 and 0.1346)
            if abs(rx) <= 0.135 and abs(ry) <= 0.507:  # MIDDLE_POCKET_Y
                in_jaw = True

            # Corner Pockets Jaw Zone (Cushions end at x=0.8147, y=0.3575)
            elif abs(rx) >= 0.814 and abs(ry) >= 0.357:
                # Must not exceed the exact pocket center to prevent void floating
                if abs(rx) <= 0.9441 and abs(ry) <= 0.4869:  # CORNER_POCKET X/Y
                    in_jaw = True

            if not in_jaw:
                return False

        return True

    def reset(self, seed=None, options=None):
        """Spawns the table state at the start of every new drill."""
        super().reset(seed=seed)

        # Reset the physics engine
        self.sim.reset()
        self.sim.in_play.fill(False)

        # 1. Place the Cue Ball FIRST (Table is empty, 100% safe)
        placement_pos_x = np.random.uniform(-self.sim.table_width / 2.0 + 0.3, self.sim.table_width / 2.0 - 0.3)
        placement_pos_y = np.random.uniform(-self.sim.table_height / 2.0 + 0.3, self.sim.table_height / 2.0 - 0.3)
        placement_pos = np.array([placement_pos_x, placement_pos_y])
        self.sim.move_cue_ball(placement_pos, baulk=False)
        self.sim.in_play[0] = True
        # self.sim.in_play[-1] = True

        # 2. Place 1 Target Ball randomly (Re-roll if it overlaps the cue ball)
        self.sim.colours[1] = self.target_color

        target_pocket = self.sim.pockets[np.random.randint(0, len(self.sim.pockets))]
        px, py = target_pocket[0], target_pocket[1]

        while True:
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(0.0, self.max_spawn_radius)

            rx = px + distance * np.cos(angle)
            ry = py + distance * np.sin(angle)

            # The geometry checker handles absolutely everything else
            if self._is_valid_spawn(rx, ry, placement_pos):
                break

        self.sim.positions[1] = [rx, ry]
        self.sim.in_play[1] = True

        # while True:
        #     bx = np.random.uniform(-self.sim.table_width / 2.0 + 0.1, self.sim.table_width / 2.0 - 0.1)
        #     by = np.random.uniform(-self.sim.table_height / 2.0 + 0.1, self.sim.table_height / 2.0 - 0.1)
        #
        #     # Check distance to Cue (0) and Red (1)
        #     dist_cb = np.hypot(bx - placement_pos[0], by - placement_pos[1])
        #     dist_red = np.hypot(bx - rx, by - ry)
        #
        #     if dist_cb > 0.06 and dist_red > 0.06:
        #         break
        #
        # self.sim.positions[-1] = [bx, by]

        # 3. Generate the observation array
        obs = self._get_obs()

        info = {}
        return obs, info

    def step(self, action):
        reward = 0.0

        vx = action[0] * 7.0
        vy = action[1] * 7.0

        speed = np.hypot(vx, vy)
        reward -= 1.5*(speed / 2)**4

        theta = action[2] * np.pi
        r = (action[3] + 1) / 2 * 0.75

        topspin = r * np.sin(theta)
        sidespin = r * np.cos(theta)
        elevation = ((action[4] + 1.0) / 2.0) * 60

        error_factor = 0
        tip_error = 0

        vx *= (1 + np.random.normal(0.0, error_factor))
        vy *= (1 + np.random.normal(0.0, error_factor))
        topspin += np.random.normal(0.0, tip_error)
        sidespin += np.random.normal(0.0, tip_error)
        elevation *= (1 + np.random.normal(0.0, error_factor))

        elevation_fraction = elevation / 60.0
        reward -= 20.0 * (elevation_fraction ** 2)

        # Penalize extreme spin (Range: 0.0 to -1.0 points)
        spin_fraction = r / 0.75
        reward -= 1.0 * (spin_fraction ** 2)

        valid = self.sim.strike_cue_ball(vx, vy, topspin, sidespin, elevation)
        if not valid:
            return self._get_obs(), -50.0, True, False, {}

        self.latest_shot_data = {
            "positions": self.sim.positions.tolist(),
            "in_play": self.sim.in_play.tolist(),
            "colours": self.sim.colours.tolist(),
            "action": action.tolist()
        }

        shot_data = self.sim.run()

        first_hit = shot_data["first_ball_hit"]
        cushion = shot_data["cushion_after_ball"]
        potted = shot_data["balls_potted"]
        oob = shot_data["error"]

        potted_black = any(self.sim.colours[b] == 3 for b in potted)

        if potted_black:
            reward -= 200.0

        if first_hit is not None and self.sim.colours[first_hit] == self.target_color:
            reward += self.REWARD_FIRST_HIT_TARGET

            # Check if ANY of the potted balls match the target color
            potted_target = any(self.sim.colours[b] == self.target_color for b in potted)

            if potted_target and 0 not in potted:
                reward += self.REWARD_POTTED
                # if not potted_black:
                #     dist_to_black = np.linalg.norm(self.sim.positions[0] - self.sim.positions[-1])
                #
                #     if dist_to_black < 0.10:
                #         # PENALTY: Frozen / too close
                #         reward -= 20.0
                #     elif 0.10 <= dist_to_black <= 0.25:
                #         # SWEET SPOT: 10cm to 25cm radius
                #         reward += 50.0
                #     else:
                #         # DECAY: Drops off exponentially the further away it stops
                #         reward += 50.0 * np.exp(-5.0 * (dist_to_black - 0.25))
            else:
                # Only calculate proximity if no target ball was potted
                target_pos = self.sim.positions[first_hit]
                min_dist = float('inf')
                for pocket in self.sim.pockets:
                    dist = np.linalg.norm(target_pos - pocket[:2])
                    if dist < min_dist:
                        min_dist = dist

                if min_dist < 0.3:
                    proximity_bonus = 50.0 * np.exp(-15.0 * min_dist)
                    reward += proximity_bonus

        if first_hit is not None and self.sim.colours[first_hit] != self.target_color:
            reward -= 50

        if 0 in potted:
            reward -= 50

        if not cushion and not potted:
            reward -= 15

        if oob:
            reward -= 40

        obs = self._get_obs()
        terminated = True
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info