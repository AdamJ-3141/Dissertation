import pygame
import numpy as np
from pool_simulation.constants import *


class Agent:
    def __init__(self):
        pass

    def get_shot_parameters(self, colours, in_play, positions) -> tuple:
        pass

    def get_cue_ball_in_hand_position(self, colours, in_play, positions) -> np.ndarray:
        pass


class Human(Agent):
    def __init__(self, renderer):
        super().__init__()
        self.renderer = renderer

        # UI State
        self.tip_x = 0.0
        self.tip_y = 0.0
        self.power = 4.0  # Base power in m/s
        self.spin_step = 0.05
        self.power_step = 0.1

    def get_cue_ball_in_hand_position(self, colours, in_play, positions) -> np.ndarray | None:
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

    def get_shot_parameters(self, colours, in_play, positions) -> tuple:
        """Pygame loop to let the user aim and set spin."""
        shot_locked = False

        self.tip_x = 0.0
        self.tip_y = 0.0
        self.power = 3.0  # Base power in m/s
        vel_x = 0.0
        vel_y = 0.0

        while not shot_locked:
            self.renderer.render(fps=60, flip=False)

            # 1. Handle Aiming (Mouse)
            mouse_pos = pygame.mouse.get_pos()

            # Draw aiming line
            self.renderer.draw_aim_line(mouse_pos[0], mouse_pos[1], self.power, self.tip_y, self.tip_x, 0.0)

            # Draw a power indicator
            self.renderer.draw_power_scale(self.power)

            # 2. Draw Spin UI
            self.renderer.draw_spin_ui(self.tip_x, self.tip_y)

            # 3. Update Display
            pygame.display.flip()
            self.renderer.clock.tick(60)

            # 4. Handle Inputs
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
                        self.power = min(MAX_SPEED, self.power + 0.2)
                    elif event.key == pygame.K_s:
                        self.power = max(0.2, self.power - 0.2)

                    # Clamp spin to a physically realistic circle
                    norm = float(np.linalg.norm([self.tip_x, self.tip_y]))
                    if norm > 0.8:
                        self.tip_x = (self.tip_x / norm) * 0.8
                        self.tip_y = (self.tip_y / norm) * 0.8

                # POWER CONTROLS (Scroll Wheel)
                elif event.type == pygame.MOUSEWHEEL:
                    self.power = np.clip(self.power + (event.y * self.power_step), 0.1, MAX_SPEED)

                # FIRE SHOT (Click)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    target_world_pos = self.renderer.screen_to_world(event.pos)
                    cue_world_pos = positions[0]
                    direction = target_world_pos - cue_world_pos

                    dist = np.linalg.norm(direction)
                    if dist > 1e-4:
                        v_hat = direction / dist
                        # Calculate final X and Y velocities based on chosen power
                        vel_x = float(v_hat[0] * self.power)
                        vel_y = float(v_hat[1] * self.power)
                        shot_locked = True

        return vel_x, vel_y, self.tip_y, self.tip_x, 0.0