import pygame
import numpy as np
import math
from pool_simulation.physics.engine import Simulation
from pool_simulation.render import Renderer


def main():
    sim = Simulation(n_obj_balls=15)
    sim.reset_to_break()
    renderer = Renderer(sim)

    def live_render(simulation_instance):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        renderer.render(fps=60)
        # We don't draw the UI during the shot to keep it clean, but you can if you want!

    running = True

    # Normalized tip position (-1.0 to 1.0)
    tip_x = 0.0  # + is Right English, - is Left English
    tip_y = 0.0  # + is Topspin, - is Backspin
    spin_step = 0.1  # How much the dot moves per key press

    print("Game Started! Use Arrow Keys to set spin. Click to shoot.")

    while running:
        # 1. Draw the base table and balls (don't flip yet!)
        renderer.render(fps=60, flip=False)

        # 2. Draw Aiming Line (if cue ball is on table)
        if sim.in_play[0]:
            mouse_pos = pygame.mouse.get_pos()
            cue_screen_pos = renderer.world_to_screen(sim.positions[0])
            pygame.draw.line(renderer.screen, (255, 255, 255), cue_screen_pos, mouse_pos, 2)

        # 3. Draw the Spin UI
        renderer.draw_spin_ui(tip_x, tip_y)

        # 4. NOW push everything to the screen
        pygame.display.flip()
        renderer.clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # --- HANDLE SPIN INPUT ---
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    tip_y += spin_step
                elif event.key == pygame.K_DOWN:
                    tip_y -= spin_step
                elif event.key == pygame.K_RIGHT:
                    tip_x += spin_step
                elif event.key == pygame.K_LEFT:
                    tip_x -= spin_step

                norm = math.hypot(tip_x, tip_y)
                if norm > 0.8:
                    tip_x = (tip_x / norm) * 0.8
                    tip_y = (tip_y / norm) * 0.8

            # --- HANDLE SHOOTING ---
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if not sim.in_play[0]:
                        continue

                    target_pos = renderer.screen_to_world(event.pos)
                    cue_pos = sim.positions[0]
                    direction = target_pos - cue_pos
                    dist = np.linalg.norm(direction)

                    if dist > 1e-4:
                        v_hat = direction / dist
                        # Let's scale speed slightly based on how far you drag the mouse!
                        # 5.0 multiplier means clicking 1 meter away = 5.0 m/s shot
                        shot_velocity = v_hat * min(8.0, dist * 5.0)

                        max_spin_rads = 40.0

                        sim.strike_cue_ball(
                            velocity_x=shot_velocity[0],
                            velocity_y=shot_velocity[1],
                            topspin=tip_y * max_spin_rads,
                            sidespin=tip_x * max_spin_rads,
                            elevation_deg=0.0
                        )

                        sim.run(framerate=60.0, frame_callback=live_render)

    pygame.quit()


if __name__ == "__main__":
    main()