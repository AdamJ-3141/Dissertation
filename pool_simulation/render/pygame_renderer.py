import pygame
import numpy as np
from pool_simulation.constants import *

COLOUR_MAP = {
    0: (200, 30, 30),  # Red
    1: (240, 240, 40),  # Yellow
    2: (20, 20, 20),  # Black
    3: (255, 255, 255)  # Cue ball
}


class Renderer:

    def __init__(self, sim, scale=PIXELS_PER_METER):
        """Renderer for the Simulation state."""
        self.sim = sim
        self.scale = scale
        self.width = int(sim.table_width * 2 * scale)
        self.height = int(sim.table_height * 2 * scale)

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("English Pool Simulation")
        self.clock = pygame.time.Clock()

    def world_to_screen(self, pos):
        """Convert world coordinates to pixel coordinates."""
        x, y = pos
        sx = self.width // 2 + x * self.scale
        sy = self.height // 2 - y * self.scale  # minus because pygame's Y grows downward
        return sx, sy

    def draw_table(self):
        green = (20, 100, 20)
        bl = self.world_to_screen((-self.sim.table_width / 2, -self.sim.table_height / 2))
        tr = self.world_to_screen((self.sim.table_width / 2, self.sim.table_height / 2))

        # unpack coords
        x1, y1 = bl
        x2, y2 = tr

        # rect expects (left, top, width, height)
        rect = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        pygame.draw.rect(self.screen, green, rect)

    def draw_balls(self):
        # Object balls
        for i in range(self.sim.n_balls):
            pos = self.sim.positions[i]
            colour = COLOUR_MAP[self.sim.colours[i]]
            pygame.draw.circle(
                self.screen,
                colour,
                self.world_to_screen(pos),
                int(self.sim.ob_radius * self.scale)
            )

        # Cue ball
        cb_colour = COLOUR_MAP[3]
        pygame.draw.circle(
            self.screen,
            cb_colour,
            self.world_to_screen(self.sim.cb_pos[0]),
            int(self.sim.cb_radius * self.scale)
        )

    def render(self, fps=60):
        """Draw the current frame."""
        self.screen.fill((0, 0, 0))  # clear
        self.draw_table()
        self.draw_balls()
        pygame.display.flip()
        self.clock.tick(fps)
