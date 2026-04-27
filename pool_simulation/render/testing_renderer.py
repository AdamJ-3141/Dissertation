import pygame
import numpy as np
from pool_simulation.physics.engine import Simulation
from pool_simulation.render.pygame_renderer import Renderer
from planner.evaluator import TableEvaluator


def main():
    sim = Simulation(n_obj_balls=15, start_break=False)

    # Clear the table
    sim.set_up_randomly(15)

    renderer = Renderer(sim)
    evaluator = TableEvaluator(sim, target_colour=1)
    nx, ny = 120, 60
    heatmap, _, _, score = evaluator.get_full_heatmap(nx=nx, ny=ny)
    print(np.sum(heatmap))
    print(score)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Call render without the target_idx
        renderer.render(fps=60, flip=False, evaluator=evaluator)

        pygame.display.flip()
        renderer.clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()