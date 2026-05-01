import pygame
import numpy as np
from pool_simulation.physics.engine import Simulation
from pool_simulation.render.pygame_renderer import Renderer
from agent import Agent
from match import Match
from planner import TableEvaluator
import time
import threading


def safe_wait(milliseconds):
    """Pauses execution while keeping the Pygame window responsive."""
    start_time = pygame.time.get_ticks()
    clock = pygame.time.Clock()

    while pygame.time.get_ticks() - start_time < milliseconds:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()  # Immediately kill the script if closed
        clock.tick(60)  # Don't hog the CPU


def main():
    # Initialize Physics Engine
    sim = Simulation(n_obj_balls=15)
    sim.set_up_randomly(15)

    # Initialize Agent with controllable weights
    agent = Agent(sim, weights=None)

    # Initialize Match and Renderer
    match = Match(sim, custom_setup=True)
    renderer = Renderer(sim)

    # Set the target color (e.g., Reds = 1)
    target_color = 1
    match.player_colours[0] = target_color

    evaluator = TableEvaluator(sim, target_color, weights=agent.weights)

    # Draw the table so it isn't black while the agent "thinks"
    renderer.render(flip=True)
    pygame.display.set_caption("Agent is thinking...")

    start_time = time.time()

    # We use a list to extract the result from the background thread
    shot_result = []

    def plan_shot():
        params = agent.get_shot_parameters(
            sim.colours, sim.in_play, sim.positions,
            target_color, match.turn_state, renderer=None
        )
        shot_result.append(params)

    # Start the agent in the background
    think_thread = threading.Thread(target=plan_shot)
    think_thread.start()

    # Keep Pygame alive while the thread runs
    clock = pygame.time.Clock()
    while think_thread.is_alive():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return  # Exit cleanly if closed mid-thought

        # Limit the while-loop to 30 FPS so it doesn't eat CPU cycles
        clock.tick(30)

    think_thread.join()
    shot_params = shot_result[0]

    print(f"Planning took: {time.time() - start_time:.4f} seconds")

    pygame.display.set_caption("English Pool Simulation")

    vx, vy, topspin, sidespin, elevation = shot_params

    if vx is not None:
        power = np.hypot(vx, vy)
        aim_angle = np.atan2(vy, vx)

        # Render table and UI elements
        renderer.render(flip=False)
        renderer.draw_spin_ui(sidespin, topspin)
        renderer.draw_power_scale(power)
        renderer.draw_elevation_ui(elevation)

        # Draw the Agent's Aim Line
        if power > 0.01:
            # Grab the cue ball's physical position
            cb_pos = sim.positions[0][:2]

            # Project an aim coordinate along the planned velocity vector
            aim_world_x = cb_pos[0] + (vx / power)
            aim_world_y = cb_pos[1] + (vy / power)

            # Convert the physical coordinates to screen pixels
            aim_screen = renderer.world_to_screen([aim_world_x, aim_world_y])

            # Draw using custom renderer method
            renderer.draw_aim_line(
                aim_screen[0], aim_screen[1],
                power, topspin, sidespin, elevation
            )

        # Display the setup to the screen
        pygame.display.flip()

        # Pause for 5 seconds to review the aim line
        pygame.time.wait(5000)

        aim_deg = np.degrees(aim_angle)
        print(f"Agent executing shot: Pwr={power:.2f}, Aim={aim_deg:.2f}°, Spin=({topspin:.2f}, {sidespin:.2f})")

        # The Recorder
        recorded_frames = []

        def record_frame_callback(state_ignored):
            # Pump the event queue so the window doesn't freeze while calculating
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            # Save a deep copy of the positions and in-play status
            # (We must copy, otherwise Pygame will just reference the final resting state)
            recorded_frames.append({
                "positions": np.copy(sim.positions),
                "in_play": np.copy(sim.in_play)
            })

        # Run the physics silently (no rendering)
        sim.strike_cue_ball(vx, vy, topspin, sidespin, elevation)
        print("Calculating physics...")
        sim.run(framerate=60, frame_callback=record_frame_callback)
        print(f"Recorded {len(recorded_frames)} frames. Starting playback...")

        # The Playback Loop
        clock = pygame.time.Clock()
        for frame in recorded_frames:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            # Overwrite the simulation's state with the recorded frame
            sim.positions = frame["positions"]
            sim.in_play = frame["in_play"]

            # Render the frozen frame
            renderer.render(flip=True)

            # Force playback at exactly 60 Frames Per Second
            clock.tick(60)

        # Pause for 5 seconds to review the final table state
        safe_wait(5000)
        running = False  # End the loop cleanly

    else:
        print("Agent found no viable shots.")
        running = False

    # Clean shutdown
    pygame.quit()


if __name__ == '__main__':
    main()