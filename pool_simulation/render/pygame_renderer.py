import pygame
import numpy as np
from pool_simulation.constants import *

COLOUR_MAP = {
    0: (200, 30, 30),  # Red Ball
    1: (240, 240, 40),  # Yellow Ball
    2: (20, 20, 20),  # Black Ball
    3: (255, 255, 255)  # Cue ball
}

WORLD_COLOURS = {
    "Cloth": (12, 196, 26),
    "Cushion": (5, 150, 16),
    "Wood": (133, 77, 0),
    "Border": (79, 46, 0),
    "Black": (0, 0, 0),
    "Grey": (40, 40, 40)
}


def draw_circular_arc(surface, color, center, radius, start_angle, end_angle, width=1):
    """
    Draws a circular arc using pygame.draw.arc.

    surface: pygame.Surface to draw on
    color: RGB tuple
    center: (x, y) center of the circle
    radius: radius of the circle
    start_angle: start angle in radians
    end_angle: end angle in radians
    width: line thickness
    """
    cx, cy = center
    rect = pygame.Rect(cx - radius, cy - radius, 2 * radius, 2 * radius)
    pygame.draw.arc(surface, color, rect, start_angle, end_angle, width)


class Renderer:

    def __init__(self, sim, scale=PIXELS_PER_METER):
        """Renderer for the Simulation state."""
        self.sim = sim
        self.scale = scale
        self.render_scale = 8
        self.width = int(sim.table_width * 2 * scale)
        self.height = int(sim.table_height * 2 * scale)

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.table = self.draw_table()
        pygame.display.set_caption("English Pool Simulation")
        self.clock = pygame.time.Clock()

    def world_to_screen(self, pos, screen_scale=1):
        """Convert world coordinates to pixel coordinates."""
        x, y = pos
        sx = self.width // 2 + x * self.scale
        sy = self.height // 2 - y * self.scale  # minus because pygame's Y grows downward
        return sx * screen_scale, sy * screen_scale

    def draw_table(self):
        line_width = 1 * self.render_scale
        table_surface = pygame.Surface((self.width * self.render_scale, self.height * self.render_scale))
        table_surface.fill(WORLD_COLOURS["Grey"])

        x1, y1 = self.world_to_screen(((-self.sim.table_width / 2 - CUSHION_WIDTH),
                                       (-self.sim.table_height / 2 - CUSHION_WIDTH)),
                                      screen_scale=self.render_scale)
        x2, y2 = self.world_to_screen(((self.sim.table_width / 2 + CUSHION_WIDTH),
                                       (self.sim.table_height / 2 + CUSHION_WIDTH)),
                                      screen_scale=self.render_scale)

        # rect expects (left, top, width, height)
        rect = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        pygame.draw.rect(table_surface, WORLD_COLOURS["Cloth"], rect)

        # left/right cushions
        arc_intervals = {
            (1, 1): ((np.pi - ALPHA_C), np.pi),
            (-1, 1): (0, ALPHA_C),
            (1, -1): (np.pi, (np.pi + ALPHA_C)),
            (-1, -1): (2 * np.pi - ALPHA_C, 2 * np.pi)
        }
        for signx in [1, -1]:
            # draw flat cushion
            rx1, ry1 = self.world_to_screen((signx * (TABLE_WIDTH / 2), -0.3575),
                                            screen_scale=self.render_scale)
            rx2, ry2 = self.world_to_screen((signx * (TABLE_WIDTH / 2 + CUSHION_WIDTH), 0.3575),
                                            screen_scale=self.render_scale)
            rect = (min(rx1, rx2), min(ry1, ry2), abs(rx2 - rx1), abs(ry2 - ry1))
            pygame.draw.rect(table_surface, WORLD_COLOURS["Cushion"], rect)
            pygame.draw.line(table_surface, WORLD_COLOURS["Black"], (rx1, ry1), (rx1, ry2), width=line_width + 1)
            for signy in [1, -1]:
                # jaw arc
                cx, cy = self.world_to_screen((signx * 1.0044, signy * 0.3575),
                                              screen_scale=self.render_scale)
                pygame.draw.circle(table_surface, WORLD_COLOURS["Cushion"],
                                   (int(cx), int(cy)),
                                   int(0.090 * self.scale * self.render_scale))
                draw_circular_arc(table_surface, WORLD_COLOURS["Black"], (int(cx), int(cy)),
                                  int(0.090 * self.scale * self.render_scale),
                                  *arc_intervals[(signx, signy)],
                                  width=line_width)
                # flat jaw
                tx1, ty1 = self.world_to_screen((signx * 0.932464, signy * 0.41158522815),
                                                screen_scale=self.render_scale)
                tx2, ty2 = self.world_to_screen((signx * 0.9744, signy * 0.467362160211),
                                                screen_scale=self.render_scale)
                pygame.draw.polygon(table_surface, WORLD_COLOURS["Cushion"], ((tx1, ty1), (tx2, ty2), (tx2, ty1)))
                pygame.draw.line(table_surface, WORLD_COLOURS["Black"], (tx1, ty1), (tx2, ty2), width=line_width)

        arc_intervals = {
            # ( 0(corner)|1(middle), signx, signy )
            (0, 1, 1): (3 * np.pi / 2, 3 * np.pi / 2 + ALPHA_C),
            (0, -1, 1): (3 * np.pi / 2 - ALPHA_C, 3 * np.pi / 2),
            (0, 1, -1): (np.pi / 2 - ALPHA_C, np.pi / 2),
            (0, -1, -1): (np.pi / 2, np.pi / 2 + ALPHA_C),
            (1, 1, 1): (3 * np.pi / 2 - ALPHA_M, 3 * np.pi / 2),
            (1, -1, 1): (3 * np.pi / 2, 3 * np.pi / 2 + ALPHA_M),
            (1, 1, -1): (np.pi / 2, np.pi / 2 + ALPHA_M),
            (1, -1, -1): (np.pi / 2 - ALPHA_M, np.pi / 2),
        }
        for signx in [1, -1]:
            for signy in [1, -1]:
                # Corner arcs
                cx, cy = self.world_to_screen((signx * 0.8147, signy * 0.5472),
                                              screen_scale=self.render_scale)
                pygame.draw.circle(table_surface, WORLD_COLOURS["Cushion"],
                                   (int(cx), int(cy)),
                                   int(0.090 * self.scale * self.render_scale))
                draw_circular_arc(table_surface, WORLD_COLOURS["Black"], (int(cx), int(cy)),
                                  int(0.090 * self.scale * self.render_scale),
                                  *arc_intervals[(0, signx, signy)],
                                  width=line_width)
                # Flat cushion
                rx1, ry1 = self.world_to_screen((signx * 0.1346, signy * (TABLE_HEIGHT / 2)),
                                                screen_scale=self.render_scale)
                rx2, ry2 = self.world_to_screen((signx * 0.8147, signy * (TABLE_HEIGHT / 2 + CUSHION_WIDTH)),
                                                screen_scale=self.render_scale)
                rect = (min(rx1, rx2), min(ry1, ry2), abs(rx2 - rx1), abs(ry2 - ry1))
                pygame.draw.rect(table_surface, WORLD_COLOURS["Cushion"], rect)
                pygame.draw.line(table_surface, WORLD_COLOURS["Black"], (rx1, ry1), (rx2, ry1), width=line_width + 1)

                # Flat corner jaw
                tx1, ty1 = self.world_to_screen((signx * 0.92456216, signy * 0.5172),
                                                screen_scale=self.render_scale)
                tx2, ty2 = self.world_to_screen((signx * 0.868785228, signy * 0.475264),
                                                screen_scale=self.render_scale)
                pygame.draw.polygon(table_surface, WORLD_COLOURS["Cushion"], ((tx1, ty1), (tx2, ty2), (tx2, ty1)))
                pygame.draw.line(table_surface, WORLD_COLOURS["Black"], (tx1, ty1), (tx2, ty2), width=line_width)

                # Middle Arcs
                cx, cy = self.world_to_screen((signx * 0.1346, signy * 0.5322),
                                              screen_scale=self.render_scale)
                pygame.draw.circle(table_surface, WORLD_COLOURS["Cushion"],
                                   (int(cx), int(cy)),
                                   int(0.075 * self.scale * self.render_scale))
                draw_circular_arc(table_surface, WORLD_COLOURS["Black"], (int(cx), int(cy)),
                                  int(0.075 * self.scale * self.render_scale),
                                  *arc_intervals[(1, signx, signy)],
                                  width=line_width)

                # Flat middle jaw
                tx1, ty1 = self.world_to_screen((signx * 0.0489767757, signy * 0.5172),
                                                screen_scale=self.render_scale)
                tx2, ty2 = self.world_to_screen((signx * 0.077404256, signy * 0.483685601),
                                                screen_scale=self.render_scale)
                pygame.draw.polygon(table_surface, WORLD_COLOURS["Cushion"], ((tx1, ty1), (tx2, ty2), (tx2, ty1)))
                pygame.draw.line(table_surface, WORLD_COLOURS["Black"], (tx1, ty1), (tx2, ty2), width=line_width)

        # Outer wood
        x1, y1 = self.world_to_screen(((-self.sim.table_width / 2 - 3 * CUSHION_WIDTH),
                                       (-self.sim.table_height / 2 - 3 * CUSHION_WIDTH)),
                                      screen_scale=self.render_scale)
        x2, y2 = self.world_to_screen(((self.sim.table_width / 2 + 3 * CUSHION_WIDTH),
                                       (self.sim.table_height / 2 + 3 * CUSHION_WIDTH)),
                                      screen_scale=self.render_scale)
        rect = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        pygame.draw.rect(table_surface, WORLD_COLOURS["Wood"], rect,
                         width=int(2 * CUSHION_WIDTH * self.render_scale * self.scale))

        # Pockets
        for signy in [1, -1]:
            cx, cy = self.world_to_screen((0, signy * 0.507),
                                          screen_scale=self.render_scale)
            pygame.draw.circle(table_surface, WORLD_COLOURS["Black"],
                               (int(cx), int(cy)),
                               int(POCKET_RADIUS * self.scale * self.render_scale))
            for signx in [1, -1]:
                cx, cy = self.world_to_screen((signx * 0.9441, signy * 0.4869),
                                              screen_scale=self.render_scale)
                pygame.draw.circle(table_surface, WORLD_COLOURS["Black"],
                                   (int(cx), int(cy)),
                                   int(POCKET_RADIUS * self.scale * self.render_scale))

        # Outline
        pygame.draw.rect(table_surface, WORLD_COLOURS["Border"], rect,
                         width=int(0.01 * line_width * self.scale))

        # Outer Cloth Black Line
        x1, y1 = self.world_to_screen(((-self.sim.table_width / 2 - CUSHION_WIDTH),
                                       (-self.sim.table_height / 2 - CUSHION_WIDTH)),
                                      screen_scale=self.render_scale)
        x2, y2 = self.world_to_screen(((self.sim.table_width / 2 + CUSHION_WIDTH),
                                       (self.sim.table_height / 2 + CUSHION_WIDTH)),
                                      screen_scale=self.render_scale)

        # rect expects (left, top, width, height)
        rect = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        pygame.draw.rect(table_surface, WORLD_COLOURS["Black"], rect, width=int(0.005 * line_width * self.scale))

        table_small = pygame.transform.smoothscale_by(table_surface, 1 / self.render_scale)
        return table_small

    def draw_balls(self):
        balls_surface = pygame.Surface(
            (self.width * self.render_scale, self.height * self.render_scale), pygame.SRCALPHA)
        # Object balls
        for i in range(self.sim.n_balls):
            pos = self.sim.positions[i]
            colour = COLOUR_MAP[self.sim.colours[i]]
            pygame.draw.circle(
                balls_surface,
                colour,
                self.world_to_screen(pos, screen_scale=self.render_scale),
                int(self.sim.ob_radius * self.scale * self.render_scale)
            )

        # Cue ball
        cb_colour = COLOUR_MAP[3]
        pygame.draw.circle(
            balls_surface,
            cb_colour,
            self.world_to_screen(self.sim.cb_pos[0], screen_scale=self.render_scale),
            int(self.sim.cb_radius * self.scale * self.render_scale)
        )
        balls_small = pygame.transform.smoothscale_by(balls_surface, 1 / self.render_scale)
        self.screen.blit(balls_small, (0, 0))

    def render(self, fps=60):
        """Draw the current frame."""
        self.screen.fill((40, 40, 40))  # clear
        self.screen.blit(self.table, (0, 0))
        self.draw_balls()
        pygame.display.flip()
        self.clock.tick(fps)
