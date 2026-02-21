import pygame
import numpy as np
from physics.engine import Simulation
from render.pygame_renderer import Renderer
from constants import *

# ---- TEMPORARY SHOT SETTINGS ----
# Cue ball initial velocity (m/s)
CUE_VELOCITY_INIT = np.array([1, 0])   # x and y components
# Cue ball angular velocity (rad/s)
CUE_ANGULAR_INIT = np.array([0.0, 0.0, 0.0])  # ωx, ωy, ωz

FPS = 90
FRAME_TIME = 1.0 / FPS


def main():
    sim = Simulation(
        n_balls=1, dt_max=FRAME_TIME)

    sim.reset(
        positions=np.array([
            [-0.6, 0.3],
            [-0.6, 0.2],
        ]),
        colours=np.array([3, 0]),
        in_play=np.array([False, True])
    )
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
                    # sim.velocities[0] = CUE_VELOCITY_INIT.copy()
                    # sim.angular[0] = CUE_ANGULAR_INIT.copy()
                    sim.velocities[1:] = np.array([
                        [2, 0],
                    ])
                    shot_played = True

        # advance physics
        dt = sim.time_step()
        accum_render_time += dt

        # render only when enough sim time has passed
        if accum_render_time >= FRAME_TIME:
            renderer.render()
            accum_render_time -= FRAME_TIME


def headless_test():
    sim = Simulation(n_balls=1, dt_max=FRAME_TIME)
    sim.reset(
        positions=np.array([
            [-1, -1],
            [0, 0],
        ]),
        colours=np.array([3, 0]),
        in_play=np.array([False, True])
    )
    sim.velocities[1] = np.array([
        [1, 1],
    ])
    sim.angular[1] = np.array([[130, -60, 0]])
    print(sim.predict_slide_roll_events())



if __name__ == "__main__":
    headless_test()
