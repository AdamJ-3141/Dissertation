import pygame
import pygame.gfxdraw
import numpy as np
from pool_simulation.constants import *
import math

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


def _value_to_color(val_norm):
    if val_norm <= 0.05:
        return (0, 0, 0, 0)
    if val_norm < 0.5:
        f = (val_norm - 0.05) / 0.45
        g = int(255 - (127 * f))
        a = int(60 + (80 * f))
        return (255, g, 0, a)
    else:
        f = (val_norm - 0.5) / 0.5
        g = int(128 - (128 * f))
        a = int(140 + (115 * f))
        return (255, g, 0, a)


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

        self.ghost_sprite = self.ball_sprites[0].copy()
        self.ghost_sprite.fill((255, 255, 255, 128), special_flags=pygame.BLEND_RGBA_MULT)

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

        # baulk line
        bx1, by1 = self.world_to_screen((-TABLE_WIDTH / 2 + TABLE_WIDTH / 5, TABLE_HEIGHT / 2),
                                        screen_scale=self.render_scale)
        bx2, by2 = self.world_to_screen((-TABLE_WIDTH / 2 + TABLE_WIDTH / 5, -TABLE_HEIGHT / 2),
                                        screen_scale=self.render_scale)

        # Calculate width, ensuring it is at least 1 pixel thick so Pygame actually renders it
        baulk_width = int(line_width)

        pygame.draw.circle(table_surface, WORLD_COLOURS["Black"],
                           self.world_to_screen((BLACK_SPOT_X, 0), screen_scale=self.render_scale),
                           2*line_width, width=3*line_width)

        pygame.draw.line(table_surface, WORLD_COLOURS["Black"],
                         (bx1, by1), (bx2, by2),
                         width=baulk_width)

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

    def draw_aim_line(self, aim_x: float, aim_y: float, speed, top, side, elevation):
        full_aim = np.array(self.screen_to_world((aim_x, aim_y)), dtype=np.float64) - self.sim.positions[0]
        if np.linalg.norm(full_aim) < 1e-5:
            return
        unit_aim = full_aim / np.linalg.norm(full_aim)
        vel_vector = unit_aim * speed
        positions = self.sim.map_to_first_coll(*vel_vector, top, side, elevation)
        if positions[-1][0] > 100: # potted
            positions = positions[:-1]
        screen_positions = list(map(self.world_to_screen, positions))
        screen_positions.insert(0, self.world_to_screen(self.sim.positions[0]))
        if len(screen_positions) >= 2:
            pygame.draw.lines(self.screen, (180, 180, 180), False, screen_positions, width=2)
        ghost_pos = self.world_to_screen(positions[-1])
        rect = self.ghost_sprite.get_rect(center=ghost_pos)
        self.screen.blit(self.ghost_sprite, rect)

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

    def draw_power_scale(self, speed: float):
        ui_margin = 30
        width = 10
        full_height = 100
        r_border = pygame.Rect(ui_margin, ui_margin, width, full_height)
        max_fill_height = full_height - 4
        current_fill_height = max_fill_height * (speed / MAX_SPEED)
        start_y = (ui_margin + full_height - 2) - current_fill_height
        r_inner = pygame.Rect(ui_margin + 2, start_y, width - 4, current_fill_height)
        pygame.draw.rect(self.screen, (180, 180, 180), r_border, width=3)
        pygame.draw.rect(self.screen, (180, 255, 180), r_inner, width=0)

    def draw_elevation_ui(self, elevation_deg: float):

        spin_ui_center_x = self.width - 60
        spin_ui_center_y = 60

        pivot_x = spin_ui_center_x - 65
        pivot_y = spin_ui_center_y

        # Cue stick segment lengths
        cue_length = 80
        tip_length = 4
        ferrule_length = 10

        # Cue stick widths (creating a nice taper)
        tip_w = 3.0
        ferrule_w = 4.0
        butt_w = 7.0

        theta = math.radians(elevation_deg)
        dx = -math.cos(theta)
        dy = -math.sin(theta)

        # The perpendicular "normal" vector to calculate the width of the stick
        nx = -dy
        ny = dx

        # Helper function to get the top/bottom vertices at any point along the stick
        def get_corners(cx, cy, width):
            half_w = width / 2.0
            return (
                (cx + nx * half_w, cy + ny * half_w),
                (cx - nx * half_w, cy - ny * half_w)
            )

        p0_x, p0_y = pivot_x, pivot_y  # Tip end
        p1_x, p1_y = p0_x + dx * tip_length, p0_y + dy * tip_length  # Ferrule start
        p2_x, p2_y = p1_x + dx * ferrule_length, p1_y + dy * ferrule_length  # Shaft start
        p3_x, p3_y = p0_x + dx * cue_length, p0_y + dy * cue_length  # Butt end

        # Generate the vertices for each cross-section
        t0_a, t0_b = get_corners(p0_x, p0_y, tip_w)
        t1_a, t1_b = get_corners(p1_x, p1_y, tip_w)
        t2_a, t2_b = get_corners(p2_x, p2_y, ferrule_w)
        t3_a, t3_b = get_corners(p3_x, p3_y, butt_w)


        pygame.draw.line(self.screen, (100, 100, 100), (pivot_x - 30, pivot_y), (pivot_x + 10, pivot_y), 1)

        pygame.draw.circle(self.screen, (60, 60, 60), (int(pivot_x + 15), int(pivot_y)), 15, 1)

        # Draw the Blue chalked tip
        pygame.draw.polygon(self.screen, (0, 150, 255), [t0_a, t1_a, t1_b, t0_b])
        # Draw the White ferrule
        pygame.draw.polygon(self.screen, (230, 230, 230), [t1_a, t2_a, t2_b, t1_b])
        # Draw the Wood shaft
        pygame.draw.polygon(self.screen, (205, 133, 63), [t2_a, t3_a, t3_b, t2_b])

        # Draw a sharp dark outline around the whole cue stick
        cue_outline = [t0_a, t1_a, t2_a, t3_a, t3_b, t2_b, t1_b, t0_b]
        pygame.draw.polygon(self.screen, (30, 30, 30), cue_outline, 1)

    def draw_heatmap_overlay(self, evaluator):
        import math
        nx, ny = 120, 60
        heatmap, playable_w, playable_h, _ = evaluator.get_full_heatmap(nx=nx, ny=ny)
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        cell_w_world = (2 * playable_w) / (nx - 1) if nx > 1 else 0
        cell_h_world = (2 * playable_h) / (ny - 1) if ny > 1 else 0

        cell_w_px = max(1, math.ceil(cell_w_world * self.scale))
        cell_h_px = max(1, math.ceil(cell_h_world * self.scale))

        x_vals = np.linspace(-playable_w, playable_w, nx)
        y_vals = np.linspace(-playable_h, playable_h, ny)

        max_val = np.max(heatmap)
        if max_val <= 0: return

        for i in range(ny):
            for j in range(nx):
                val = heatmap[i, j]
                if val <= 0: continue

                val_norm = val / max_val
                color = _value_to_color(val_norm)

                if color[3] > 0:
                    world_x = x_vals[j] - (cell_w_world / 2)
                    world_y = y_vals[i] + (cell_h_world / 2)

                    px_x, px_y = self.world_to_screen((world_x, world_y))

                    rect = pygame.Rect(int(px_x), int(px_y), cell_w_px, cell_h_px)
                    pygame.draw.rect(overlay, color, rect)

        self.screen.blit(overlay, (0, 0))

    def render(self, fps=60, flip=True, debug_shots=None, evaluator=None):
        """Draw the current frame."""
        self.screen.fill((40, 40, 40))  # clear
        self.screen.blit(self.table, (0, 0))

        if evaluator:
            self.draw_heatmap_overlay(evaluator)

        if debug_shots:
            radius_px = int(self.sim.ob_radius * self.scale)

            for shot in debug_shots:
                cb_pos = self.sim.positions[0]
                ob_pos = self.sim.positions[shot["target_idx"]]
                gb_pos = shot["ghost_ball_pos"]
                target_pt = shot["target_pt"]
                shot_type = shot.get("type", "")

                cb_px = self.world_to_screen(cb_pos)
                ob_px = self.world_to_screen(ob_pos)
                gb_px = self.world_to_screen(gb_pos)
                target_px = self.world_to_screen(target_pt)

                cb_path = [cb_px]
                if "kick" in shot_type and "bounce_points" in shot:
                    for bp in shot["bounce_points"]:
                        cb_path.append(self.world_to_screen(bp))
                cb_path.append(gb_px)

                if len(cb_path) > 2:
                    pygame.draw.lines(self.screen, (200, 200, 200), False, cb_path, 1)
                else:
                    pygame.draw.line(self.screen, (200, 200, 200), cb_path[0], cb_path[1], 1)

                # Draw the Primary Ghost Ball (Where Cue Ball aims)
                pygame.draw.circle(self.screen, (255, 255, 255), (int(gb_px[0]), int(gb_px[1])), radius_px, 1)

                if shot_type == "plant":
                    combo_pos = self.sim.positions[shot["combo_idx"]]
                    gb1_pos = shot["gb1_pos"]

                    combo_px = self.world_to_screen(combo_pos)
                    gb1_px = self.world_to_screen(gb1_pos)

                    # Combo Ball -> GB1 (Orange)
                    pygame.draw.line(self.screen, (255, 165, 0), combo_px, gb1_px, 1)
                    # Target Ball -> Pocket (Yellow)
                    pygame.draw.line(self.screen, (255, 255, 50), ob_px, target_px, 1)

                    # Draw Intermediate Ghost Ball
                    pygame.draw.circle(self.screen, (255, 165, 0), (int(gb1_px[0]), int(gb1_px[1])), radius_px, 1)

                elif shot_type == "carom":
                    impact_pos = shot["gb1_pos"]  # Where Target Ball hits Kiss Ball
                    impact_px = self.world_to_screen(impact_pos)

                    # Target Ball -> Impact Point (Orange)
                    pygame.draw.line(self.screen, (255, 165, 0), ob_px, impact_px, 1)
                    # Impact Point -> Pocket (Yellow)
                    pygame.draw.line(self.screen, (255, 255, 50), impact_px, target_px, 1)

                    # Draw Intermediate Ghost Ball (Target ball at impact)
                    pygame.draw.circle(self.screen, (255, 165, 0), (int(impact_px[0]), int(impact_px[1])), radius_px, 1)

                else:
                    ob_path = [ob_px]
                    if "bank" in shot_type and "bounce_points" in shot:
                        for bp in shot["bounce_points"]:
                            ob_path.append(self.world_to_screen(bp))
                    ob_path.append(target_px)

                    if len(ob_path) > 2:
                        pygame.draw.lines(self.screen, (255, 255, 50), False, ob_path, 1)
                    else:
                        pygame.draw.line(self.screen, (255, 255, 50), ob_path[0], ob_path[1], 1)

        self.update_cue_ball_rotation(dt=1 / fps)

        self.draw_balls()

        if flip:
            pygame.display.flip()
            self.clock.tick(fps)

    def wait_for_space(self):
        """Blocks execution until the user presses the SPACE bar."""
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    waiting = False
            self.clock.tick(30)