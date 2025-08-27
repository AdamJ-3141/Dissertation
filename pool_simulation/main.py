import pygame
from physics.engine import Simulation
from render.pygame_renderer import Renderer
from constants import *

# ---- TEMPORARY SHOT SETTINGS ----
# Cue ball initial velocity (m/s)
CUE_VELOCITY_INIT = np.array([1.5, 0.5])   # x and y components
# Cue ball angular velocity (rad/s)
CUE_ANGULAR_INIT = np.array([0.0, 0.0, 20.0])  # ωx, ωy, ωz


def main():
    sim = Simulation(
        n_balls=2,
        table_width=TABLE_WIDTH,
        table_height=TABLE_HEIGHT,
        cb_radius=CUE_BALL_RADIUS,
        cb_mass=CUE_BALL_MASS,
        ob_radius=OBJECT_BALL_RADIUS,
        ob_mass=OBJECT_BALL_MASS,
        mu_s=MU_S,
        mu_r=MU_R)

    sim.reset(
        np.array([
            [-0.547, 0],
            [BLACK_SPOT_X, 0],
            [0, 0.05]
        ]),
        np.array([3, 2, 0]), )
    renderer = Renderer(sim)

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
                    sim.angular_velocities[0] = CUE_ANGULAR_INIT.copy()
                    shot_played = True
                    # schedule first events
                    sim.schedule_initial_events()

        if shot_played:
            # advance simulation by events
            sim.step_to_next_event()

        renderer.render()


if __name__ == "__main__":
    main()
