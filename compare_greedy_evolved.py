import json
import numpy as np
import pygame
from pool_simulation.physics.engine import Simulation
from pool_simulation.render import Renderer
from agent import Agent, GreedyAgent


def run_pygame_comparison():
    sim = Simulation(n_obj_balls=7)
    renderer = Renderer(sim)

    # Load agents
    try:
        with open('best_gen_19.json', 'r') as f:
            evolved_weights = json.load(f)
        evolved_agent = Agent(sim, evolved_weights)
        greedy_agent = GreedyAgent(sim)
    except Exception as e:
        print("Could not load agent files:", e)
        return

    def execute_and_render_shot(agent, agent_name):
        # Set up the exact table state
        sim.positions = np.array([
            [-0.581, 0.106],
            [-0.895, 0.433],
            [0.697, -0.155],
            [-0.011,0.085],
            [-0.286,0.123],
            [-0.141,0.103],
            [0.13, 0.08],
            [0.268, 0.076]
        ])
        sim.colours = np.array([0, 1, 1, 2, 2, 2, 2, 2])
        sim.in_play = np.array([True, True, True, True, True, True, True, True])

        sim.velocities.fill(0.0)
        sim.angular.fill(0.0)
        sim.ball_states.fill("STOPPED")

        # Save the initial position state[cite: 12]
        sim.save_state()

        # Get the agent's shot
        print(f"Querying {agent_name}...")
        vx, vy, t, s, el = agent.get_shot_parameters(
            sim.colours, sim.in_play, sim.positions, target_color=1, turn_state=0
        )

        # Run the shot headlessly with a callback to save positions
        trace_data = []

        def headless_callback(state):
            trace_data.append({
                "pos": state.positions.copy(),
                "in_play": state.in_play.copy()
            })

        sim.strike_cue_ball(vx, vy, t, s, el)
        sim.run(framerate=60, frame_callback=headless_callback)

        # Ensure the final resting state is captured in the trace
        headless_callback(sim)

        # Load the initial position
        sim.load_state()

        # Render lines connecting the positions points
        pygame.display.set_caption(f"{agent_name} - Trace Result. Press SPACE for next.")

        # Draw the base table with the initial frozen positions
        renderer.render(fps=60, flip=False)

        trace_colors = {
            0: (200, 200, 200),  # Cue Ball
            1: (255, 100, 100),  # Red
            2: (255, 215, 0),  # Yellow
            3: (50, 50, 50)  # Black
        }

        # Iterate through every ball and draw its path
        for b_idx in range(sim.n_obj_balls + 1):
            # Extract coordinates only for frames where the ball was on the table
            path = [frame["pos"][b_idx] for frame in trace_data if frame["in_play"][b_idx]]

            if len(path) > 1:
                # Convert world coordinates to screen coordinates
                screen_path = [renderer.world_to_screen(p) for p in path]
                color = trace_colors.get(sim.colours[b_idx], (255, 255, 255))
                line_width = 3 if b_idx == 0 else 2

                pygame.draw.lines(renderer.screen, color, False, screen_path, line_width)

        # Push the finalized composite image to the screen
        pygame.display.flip()

        # Repeat once space bar is pressed
        renderer.wait_for_space()

    # Run the sequence
    execute_and_render_shot(greedy_agent, "Greedy Agent")
    execute_and_render_shot(evolved_agent, "Evolved Agent (Gen 19)")

    pygame.quit()
    print("Comparison complete.")


if __name__ == '__main__':
    run_pygame_comparison()