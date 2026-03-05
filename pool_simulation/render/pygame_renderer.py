import pygame
import pygame.gfxdraw
import numpy as np
from pool_simulation.constants import *
import math

COLOUR_MAP = {
    0: (255, 255, 255),  # Cue ball
    1: (200, 30, 30),  # Red Ball
    2: (240, 240, 40),  # Yellow Ball
    3: (20, 20, 20),  # Black Ball

# --- Debug Colours ---
    -1: (0, 255, 255),   # Cyan
    -2: (255, 0, 255),   # Magenta
    -3: (255, 128, 0),   # Orange
    -4: (128, 0, 255),   # Neon Purple
    -5: (0, 100, 255),   # Bright Blue
    -6: (143, 255, 188)  # Light Green-Blue
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


def make_ball_sprite(float_radius, colour, render_scale):
    float_diameter = float_radius * 2.0

    big_d = int(round(float_diameter * render_scale))
    big_r = big_d // 2

    big_surf = pygame.Surface((big_d, big_d), pygame.SRCALPHA)
    pygame.draw.circle(big_surf, colour, (big_r, big_r), big_r)

    target_d = int(round(float_diameter))

    small_surf = pygame.transform.smoothscale(
        big_surf, (target_d, target_d)
    )
    return small_surf


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

        self.cue_ball_spots = np.array([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0]
        ])

        self.ball_sprites = {
            c: make_ball_sprite(
                self.sim.ob_radius * self.scale,
                COLOUR_MAP[c],
                self.render_scale
            )
            for c in COLOUR_MAP.keys() if c != 0  # without cue ball
        }
        self.ball_sprites[0] = make_ball_sprite(
                self.sim.cb_radius * self.scale,
                COLOUR_MAP[0],
                self.render_scale
            )

    def world_to_screen(self, pos, screen_scale=1):
        """Convert world coordinates to pixel coordinates."""
        x, y = pos
        sx = self.width // 2 + x * self.scale
        sy = self.height // 2 - y * self.scale  # minus because pygame's Y grows downward
        return sx * screen_scale, sy * screen_scale

    def screen_to_world(self, screen_pos):
        """Convert Pygame pixel coordinates back to physics world meters."""
        sx, sy = screen_pos
        x = (sx - self.width // 2) / self.scale
        y = (self.height // 2 - sy) / self.scale
        return np.array([x, y])

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
            cx, cy = self.world_to_screen((MIDDLE_POCKET_X, signy * MIDDLE_POCKET_Y),
                                          screen_scale=self.render_scale)
            pygame.draw.circle(table_surface, WORLD_COLOURS["Black"],
                               (int(cx), int(cy)),
                               int(POCKET_RADIUS * self.scale * self.render_scale))
            for signx in [1, -1]:
                cx, cy = self.world_to_screen((signx * CORNER_POCKET_X, signy * CORNER_POCKET_Y),
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
        for i in range(self.sim.n_obj_balls + 1):
            if self.sim.in_play[i]:
                pos = self.world_to_screen(self.sim.positions[i])
                rect = self.ball_sprites[self.sim.colours[i]].get_rect(center=pos)
                self.screen.blit(self.ball_sprites[self.sim.colours[i]], rect)
                # DRAW THE ARAMITH MEASLES ON THE CUE BALL
                if i == 0:
                    pos_screen = self.world_to_screen(self.sim.positions[0])
                    center_x = pos_screen[0]
                    center_y = pos_screen[1]

                    radius_px = self.sim.radii[0] * self.scale

                    max_spot_size = radius_px * 0.25

                    measle_color = (200, 30, 30)

                    # Check all 6 spots
                    for spot in self.cue_ball_spots:
                        if spot[2] > 0.0:
                            offset_x = spot[0] * radius_px
                            offset_y = -spot[1] * radius_px

                            spot_screen_x = center_x + offset_x
                            spot_screen_y = center_y + offset_y

                            # Foreshortening squish
                            spot_radius = max(1, int(max_spot_size * spot[2]))

                            pygame.draw.circle(
                                self.screen,
                                measle_color,
                                (int(spot_screen_x), int(spot_screen_y)),
                                spot_radius
                            )

    def draw_spin_ui(self, tip_x: float, tip_y: float):
        ui_radius = 40
        margin = 20

        center_x = self.width - ui_radius - margin
        center_y = ui_radius + margin

        pygame.draw.circle(self.screen, (240, 240, 240), (center_x, center_y), ui_radius)
        pygame.draw.circle(self.screen, (50, 50, 50), (center_x, center_y), ui_radius, width=2)
        pygame.draw.line(self.screen, (180, 180, 180), (center_x - ui_radius, center_y),
                         (center_x + ui_radius, center_y))
        pygame.draw.line(self.screen, (180, 180, 180), (center_x, center_y - ui_radius),
                         (center_x, center_y + ui_radius))

        dot_x = center_x + int(tip_x * ui_radius)
        dot_y = center_y - int(tip_y * ui_radius)
        pygame.draw.circle(self.screen, (255, 30, 30), (dot_x, dot_y), 5)

    def update_cue_ball_rotation(self, dt: float):
        """Rotates the 6 visual measle spots on the cue ball based on 3D angular velocity."""
        # Only process if the cue ball is actually on the table
        if not self.sim.in_play[0]:
            return

        w = self.sim.angular[0]
        w_norm = np.linalg.norm(w)

        # If the cue ball is spinning
        if w_norm > 1e-6:
            axis = w / w_norm
            theta = w_norm * dt

            cos_t = math.cos(theta)
            sin_t = math.sin(theta)

            # v is shape (6, 3)
            v = self.cue_ball_spots

            cross_term = np.cross(axis[None, :], v)
            dots = np.dot(v, axis).reshape(-1, 1)
            dot_term = dots * axis * (1.0 - cos_t)
            new_v = (v * cos_t) + (cross_term * sin_t) + dot_term
            norms = np.linalg.norm(new_v, axis=1, keepdims=True)
            self.cue_ball_spots = new_v / norms

    def render(self, fps=60, flip=True):
        """Draw the current frame."""
        self.screen.fill((40, 40, 40))  # clear
        self.screen.blit(self.table, (0, 0))
        self.update_cue_ball_rotation(dt=1 / fps)
        self.draw_balls()
        if flip:
            pygame.display.flip()
            self.clock.tick(fps)