import pygame
import sys
import numpy as np
import math
from pool_simulation.physics.engine import Simulation
from pool_simulation.render.pygame_renderer import Renderer
from pool_simulation.constants import *


def draw_dashed_line(surface, color, start_pos, end_pos, width=2, dash_length=10):
    """Helper to draw dashed lines for the virtual mirror trajectory."""
    x1, y1 = start_pos
    x2, y2 = end_pos
    dl = math.hypot(x2 - x1, y2 - y1)
    if dl == 0: return

    dashes = int(dl / dash_length)
    for i in range(dashes):
        # Draw every other dash segment
        if i % 2 == 0:
            start_x = x1 + (x2 - x1) * (i / dashes)
            start_y = y1 + (y2 - y1) * (i / dashes)
            end_x = x1 + (x2 - x1) * ((i + 1) / dashes)
            end_y = y1 + (y2 - y1) * ((i + 1) / dashes)
            pygame.draw.line(surface, color, (start_x, start_y), (end_x, end_y), width)


def main():
    # Initialize Engine
    sim = Simulation(n_obj_balls=4)
    sim.in_play[:] = False

    # Setup Cue Ball
    sim.positions[0] = np.array([0.32, 0.09])
    sim.in_play[0] = True

    # Target Ball (Red)
    sim.positions[1] = np.array([0.54, 0.21])
    sim.colours[1] = 2
    sim.in_play[1] = True

    # Distractor Ball
    sim.positions[2] = np.array([0.75, 0.3])
    sim.colours[2] = 1
    sim.in_play[2] = True

    renderer = Renderer(sim)

    # Calculate Effective Bounds for the mirror math
    w = (TABLE_WIDTH / 2) - CUE_BALL_RADIUS
    h = (TABLE_HEIGHT / 2) - CUE_BALL_RADIUS

    # Manually execute the Mirror Table Algorithm for the diagram
    cb_pos = sim.positions[0]
    target_pos = sim.positions[2]

    # Mirror Target across the Top Cushion
    mirrored_target = np.array([target_pos[0], 2 * h - target_pos[1]])

    # Calculate the intersection point on the Top Cushion
    dx = mirrored_target[0] - cb_pos[0]
    dy = mirrored_target[1] - cb_pos[1]
    t = (h - cb_pos[1]) / dy
    bounce_pt = np.array([cb_pos[0] + t * dx, h])

    print("\nControls:")
    print("  [S] - Save screenshot to 'mirror_table.png'")
    print("  [ESC] - Quit")

    # Pygame Display Loop
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    pygame.image.save(renderer.screen, "mirror_table.png")
                    print("Saved 'mirror_table.png' successfully!")
                elif event.key == pygame.K_ESCAPE:
                    running = False

        # Render base table
        renderer.render_shots = []
        renderer.render(flip=False)

        # Convert to screen coordinates
        cb_px = renderer.world_to_screen(cb_pos)
        bp_px = renderer.world_to_screen(bounce_pt)
        target_px = renderer.world_to_screen(target_pos)
        mirrored_px = renderer.world_to_screen(mirrored_target)

        # Draw solid real trajectory (CB -> Bounce Point -> Target)
        pygame.draw.line(renderer.screen, (255, 255, 255), cb_px, bp_px, 2)
        pygame.draw.line(renderer.screen, (255, 255, 255), bp_px, target_px, 2)

        # Draw a red dashed line showing the direct path is snookered!
        draw_dashed_line(renderer.screen, (255, 50, 50), cb_px, target_px, width=2, dash_length=6)

        # Draw transparent target ball showing the virtual mirror position
        r_ob = int(OBJECT_BALL_RADIUS * renderer.scale)
        pygame.draw.circle(renderer.screen, (150, 150, 255), (int(mirrored_px[0]), int(mirrored_px[1])), r_ob, 1)

        pygame.display.flip()
        renderer.clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()