import pygame
from physics.engine import Simulation
from render.pygame_renderer import Renderer
from constants import *

# ---- TEMPORARY SHOT SETTINGS ----
# Cue ball initial velocity (m/s)
CUE_VELOCITY_INIT = np.array([11, 0])   # x and y components
# Cue ball angular velocity (rad/s)
CUE_ANGULAR_INIT = np.array([90.0, -130.0, 0.0])  # ωx, ωy, ωz

FPS = 60
FRAME_TIME = 1.0 / FPS
DT = 0.0005


def main():
    sim = Simulation(
        n_balls=2)

    sim.reset(
        np.array([
            [-0.547, 0],
            [BLACK_SPOT_X, 0],
            [0, 0.05]
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
        sim.time_step(DT)
        accum_render_time += DT

        # render only when enough sim time has passed
        if accum_render_time >= FRAME_TIME:
            renderer.render()
            accum_render_time -= FRAME_TIME


if __name__ == "__main__":
    main()
