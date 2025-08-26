import pygame
from physics.engine import Simulation
from render.pygame_renderer import Renderer
from constants import *


def main():
    sim = Simulation(
        n_balls=2,
        table_width=TABLE_WIDTH,
        table_height=TABLE_HEIGHT,
        cb_radius=CUE_BALL_RADIUS,
        cb_mass=CUE_BALL_MASS,
        ob_radius=OBJECT_BALL_RADIUS,
        ob_mass=OBJECT_BALL_MASS)

    sim.reset(
        np.array([
            [BLACK_SPOT_X, 0],
            [0, 0.05]
        ]),
        np.array([-0.547, 0]),
        np.array([2, 0]), )
    renderer = Renderer(sim)

    running = True
    while running:
        # Handle quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Step physics
        sim.step()

        # Render
        renderer.render()


if __name__ == "__main__":
    main()
