import pygame
import sys
from pool_simulation.physics.engine import Simulation
from pool_simulation.render.pygame_renderer import Renderer
from planner.shot_generator import ShotGenerator


def main():
    # 1. Initialize Engine and generate a random table
    sim = Simulation(n_obj_balls=15)
    sim.set_up_randomly(15)
    renderer = Renderer(sim)

    # 2. Assign the AI a target colour (1 = Reds, 2 = Yellows)
    target_color = 1

    def generate_shots():
        """Helper to fetch the current table's valid shots."""
        target_set = [i for i in range(1, sim.n_obj_balls + 1)
                      if sim.colours[i] == target_color and sim.in_play[i]]

        # If no reds are left, aim for the black
        if not target_set:
            target_set = [i for i in range(1, sim.n_obj_balls + 1) if sim.colours[i] == 3 and sim.in_play[i]]

        # Run the generator
        generator = ShotGenerator(sim, target_set)
        shots = generator.get_all_shots()

        print(f"\n--- Table Setup ---")
        print(f"Target Colour: {target_color}")
        print(f"Total Candidates Found: {len(shots)}")
        print(sim.positions)
        print(sim.colours)

        # Breakdown by shot type
        types = {}
        for s in shots:
            t = s.get("type", "unknown")
            types[t] = types.get(t, 0) + 1

        for t, count in types.items():
            print(f"  - {t}: {count}")

        return shots

    # Initial generation
    candidates = generate_shots()

    # 3. Pygame Display Loop
    print("\nControls:")
    print("  [SPACE] - Randomize the balls and re-generate shots")
    print("  [1]     - Switch target to Reds")
    print("  [2]     - Switch target to Yellows")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    sim.set_up_randomly(15)
                    candidates = generate_shots()
                elif event.key == pygame.K_1:
                    target_color = 1
                    candidates = generate_shots()
                elif event.key == pygame.K_2:
                    target_color = 2
                    candidates = generate_shots()

        # Render the table, passing our candidates straight into the debug_shots argument!
        renderer.render(fps=60, flip=True, debug_shots=candidates)

    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()