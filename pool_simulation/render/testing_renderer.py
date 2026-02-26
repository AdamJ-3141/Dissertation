import pygame
from pygame_renderer import Renderer
from pool_simulation.physics import Simulation
import numpy as np

def live_render_shot():
    # 1. Setup Simulation and Renderer
    sim = Simulation(n_obj_balls=1)

    # Place Object ball at (0, 0), Cue ball at (-0.5, -0.01) to create an offset cut shot
    positions = np.array([
        [-0.5, 0.0],  # Cue Ball
        [0.0, 0.32]  # Object Ball 1
    ])
    colours = np.array([0, 2])  # 0 = White, 2 = Yellow
    in_play = np.array([True, True])

    sim.reset(positions=positions, colours=colours, in_play=in_play)

    renderer = Renderer(sim)

    # 2. Define the Pygame Hook
    def pygame_callback(simulation_instance):
        # Keep the OS happy and allow window closing
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Draw the frame
        renderer.render(fps=60)

    # Propel the cue ball with heavy bottom-left spin
    active_mask = np.zeros(sim.n_obj_balls + 1, dtype=bool)
    active_mask[0] = True

    # Velocity: 2.0 m/s to the right. Spin: massive backspin (wy) and left-spin (wz)
    sim.propel_ball(
        ball_mask=active_mask,
        velocities=np.array([[1, 0.0]]),
        angulars=np.array([[-30.0, 50.0, -20.0]])
    )

    sim.run(framerate=60.0, frame_callback=pygame_callback)

    # Optional: Keep window open after balls stop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        renderer.render(fps=60)


def replay_shot():

    sim = Simulation(n_obj_balls=6)

    sim.reset_to_six_red()

    # 1. Collect history completely headless (Takes < 0.01 seconds)
    history = []

    def record_frame(simulation_instance):
        # Save a copy of the current state of all balls
        state = {
            'positions': simulation_instance.positions.copy(),
            'in_play': simulation_instance.in_play.copy()
        }
        history.append(state)

    # Propel the cue ball with heavy bottom-left spin
    active_mask = np.zeros(sim.n_obj_balls + 1, dtype=bool)
    active_mask[0] = True

    # Velocity: 5.0 m/s to the right. Spin: massive backspin (wy) and left-spin (wz)

    sim.move_cue_ball(np.array([-0.66, 0.19]))

    sim.strike_cue_ball(3.3, -0.56, -60, 0, 0)
    t = sim.run(framerate=120.0, frame_callback=record_frame, verbose=True)

    print(f"\nTime for completion: {t}")
    # 2. Playback the recording in Pygame
    renderer = Renderer(sim)
    frame_idx = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if frame_idx < len(history):
            # Temporarily inject recorded state back into the sim for rendering
            sim.positions = history[frame_idx]['positions']
            sim.in_play = history[frame_idx]['in_play']
            frame_idx += 1

        renderer.render(fps=120)


replay_shot()