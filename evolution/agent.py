import pygame
import numpy as np
from pool_simulation.constants import *


class Agent:
    def __init__(self, sim):
        self.sim = sim

    def get_shot_parameters(self, colours: np.ndarray, in_play: np.ndarray, positions: np.ndarray,
                            target_color: int, turn_state: int) -> tuple:
        vx = None
        vy = None
        topspin = None
        sidespin = None
        elevation = None

        return vx, vy, topspin, sidespin, elevation

    def get_cue_ball_in_hand_position(self, colours: np.ndarray, in_play: np.ndarray, positions: np.ndarray,
                                      target_color: int, turn_state: int) -> np.ndarray:

        place_x = 0.0
        place_y = 0.0

        return np.array([place_x, place_y])


class Human(Agent):
    def __init__(self, sim,  renderer):
        super().__init__(sim)

        self.renderer = renderer

        # UI State
        self.tip_x = 0.0
        self.tip_y = 0.0
        self.power = 4.0  # Base power in m/s
        self.elevation = 0.0
        self.spin_step = 0.05
        self.power_step = 0.1
        self.elevation_step = 2

    def get_cue_ball_in_hand_position(self, colours, in_play, positions,
                                      target_colour, turn_state) -> np.ndarray | None:
        """Pygame loop to let the user drag the cue ball."""
        placed = False

        while not placed:
            self.renderer.render(fps=60, flip=False)

            # Draw a glowing cue ball at the mouse cursor
            mouse_pos = pygame.mouse.get_pos()
            pygame.draw.circle(self.renderer.screen, (200, 255, 200), mouse_pos,
                               int(OBJECT_BALL_RADIUS * self.renderer.scale))

            pygame.display.flip()
            self.renderer.clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # User clicked! Convert screen coords back to world coords
                    world_pos = self.renderer.screen_to_world(event.pos)
                    return np.array(world_pos)
        return None

    def get_shot_parameters(self, colours, in_play, positions, target_colour, turn_state) -> tuple:
        """Pygame loop to let the user aim and set spin."""
        shot_locked = False

        # Clear any leftover clicks from the ball-in-hand phase
        pygame.event.clear()

        self.tip_x = 0.0
        self.tip_y = 0.0
        self.power = 8.0 if self.sim.is_break else 2.0 # Base power in m/s
        self.elevation = 0.0  # This now acts as the player's "Manual Override"

        vel_x = 0.0
        vel_y = 0.0
        actual_elevation = 0.0

        while not shot_locked:
            self.renderer.render(fps=60, flip=False)

            # 1. Handle Aiming (Mouse)
            mouse_pos = pygame.mouse.get_pos()
            target_world_pos = self.renderer.screen_to_world(mouse_pos)
            cue_world_pos = positions[0]
            direction = target_world_pos - cue_world_pos
            dist = np.linalg.norm(direction)

            # Calculate the theoretical velocities based on the mouse position
            test_vx, test_vy = 0.0, 0.0
            if dist > 1e-4:
                v_hat = direction / dist
                test_vx = float(v_hat[0] * self.power)
                test_vy = float(v_hat[1] * self.power)

            min_el = 89.0  # Fallback to maximum elevation
            if dist > 1e-4:
                # Ask the engine to find the lowest valid angle for this exact shot
                for test_e in range(0, 90):
                    if self.sim.validate_shot(test_vx, test_vy, self.tip_y, self.tip_x, float(test_e)):
                        min_el = float(test_e)
                        break

            # The final elevation is whichever is higher: the required physical minimum, or the player's manual input
            actual_elevation = max(self.elevation, min_el)

            # Double check that the final calculated shot is legal
            is_valid = False
            if dist > 1e-4:
                is_valid = self.sim.validate_shot(test_vx, test_vy, self.tip_y, self.tip_x, actual_elevation)

            # ==========================================
            # RENDER THE UI
            # ==========================================
            # Only draw the aiming line if the shot is physically possible!
            if is_valid:
                self.renderer.draw_aim_line(mouse_pos[0], mouse_pos[1], self.power, self.tip_y, self.tip_x,
                                            actual_elevation)

            self.renderer.draw_power_scale(self.power)
            self.renderer.draw_spin_ui(self.tip_x, self.tip_y)

            # Pass our dynamically calculated elevation to the UI so the player sees the cue raise!
            self.renderer.draw_elevation_ui(actual_elevation)

            pygame.display.flip()
            self.renderer.clock.tick(60)

            # ==========================================
            # HANDLE INPUTS
            # ==========================================
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

                # SPIN CONTROLS (Arrow Keys)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.tip_y += self.spin_step
                    elif event.key == pygame.K_DOWN:
                        self.tip_y -= self.spin_step
                    elif event.key == pygame.K_RIGHT:
                        self.tip_x += self.spin_step
                    elif event.key == pygame.K_LEFT:
                        self.tip_x -= self.spin_step

                    # POWER CONTROLS (W / S)
                    elif event.key == pygame.K_w:
                        self.power = min(MAX_SPEED, self.power + self.power_step)
                    elif event.key == pygame.K_s:
                        self.power = max(0.2, self.power - self.power_step)

                    # Clamp spin to a physically realistic circle
                    norm = float(np.linalg.norm([self.tip_x, self.tip_y]))
                    if norm > 0.8:
                        self.tip_x = (self.tip_x / norm) * 0.8
                        self.tip_y = (self.tip_y / norm) * 0.8

                    # ELEVATION CONTROLS (E / D) - This now acts as a manual floor
                    elif event.key == pygame.K_e:
                        self.elevation = min(89.0, self.elevation + self.elevation_step)
                    elif event.key == pygame.K_d:
                        self.elevation = max(0.0, self.elevation - self.elevation_step)

                # POWER CONTROLS (Scroll Wheel)
                elif event.type == pygame.MOUSEWHEEL:
                    self.power = np.clip(self.power + (event.y * self.power_step), 0.1, MAX_SPEED)

                # FIRE SHOT (Click)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # ONLY allow the shot to fire if it cleared the validation check!
                    if is_valid:
                        vel_x = test_vx
                        vel_y = test_vy
                        shot_locked = True

        return vel_x, vel_y, self.tip_y, self.tip_x, actual_elevation