import pygame
import numpy as np
from physics.engine import Simulation
from render.pygame_renderer import Renderer
from constants import *

# ---- TEMPORARY SHOT SETTINGS ----
# Cue ball initial velocity (m/s)
CUE_VELOCITY_INIT = np.array([2, -0.2])   # x and y components
# Cue ball angular velocity (rad/s)
CUE_ANGULAR_INIT = np.array([0.0, 0.0, 0.0])  # ωx, ωy, ωz

FPS = 90
FRAME_TIME = 1.0 / FPS


def main():
    sim = Simulation(
        n_balls=2, dt_max=FRAME_TIME/2)

    sim.reset(
        np.array([
            [0.22, -0.333],
            [BLACK_SPOT_X, 0],
            [0.25, 0.05]
        ]),
        np.array([3, 2, 0]), )
    renderer = Renderer(sim)

    accum_render_time = 0.0
    shot_played = False
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not shot_played:
                    # apply initial velocity + spin to cue ball
                    sim.velocities[0] = CUE_VELOCITY_INIT.copy()
                    sim.angular[0] = CUE_ANGULAR_INIT.copy()
                    shot_played = True

        # advance physics
        dt = sim.time_step()
        accum_render_time += dt

        # render only when enough sim time has passed
        if accum_render_time >= FRAME_TIME:
            renderer.render()
            accum_render_time -= FRAME_TIME


if __name__ == "__main__":
    main()
