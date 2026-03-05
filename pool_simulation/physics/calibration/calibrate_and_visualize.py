import json
import pygame
import numpy as np
from scipy.optimize import least_squares
from pool_simulation.physics.engine import Simulation
from pool_simulation.render.pygame_renderer import Renderer


def load_shot_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    t0 = data[0]['time']
    for d in data:
        d['time'] -= t0
    return data


def get_base_kinematics(data):
    start_pos = np.array([data[0]['x'], data[0]['y']])
    dt = data[1]['time'] - data[0]['time']
    dx = data[1]['x'] - data[0]['x']
    dy = data[1]['y'] - data[0]['y']
    return start_pos, np.array([dx / dt, dy / dt])


def simulate_trajectory(physics_params, start_pos, velocity):
    mu_s, mu_r, e_c, mu_c = physics_params
    sim = Simulation(n_obj_balls=0, mu_s=mu_s, mu_r=mu_r, e_c=e_c, mu_c=mu_c, k_n=1e4)
    sim.positions[0] = start_pos
    sim.in_play[0] = True
    sim.strike_cue_ball(velocity_x=velocity[0], velocity_y=velocity[1], topspin=0.0, sidespin=0.0)

    history = []

    def record_frame(s):
        history.append(s.positions[0].copy())

    sim.run(framerate=60.0, frame_callback=record_frame)
    return np.array(history)


def objective_function(params, shot1_data, shot3_data, base_start1, base_start3):
    physics_params = params[0:4]
    v1 = params[4:6]
    v3 = params[6:8]
    residuals = []

    for data, start_pos, vel in [(shot1_data, base_start1, v1), (shot3_data, base_start3, v3)]:
        sim_history = simulate_trajectory(physics_params, start_pos, vel)
        sim_frames = len(sim_history)

        for pt in data:
            frame_idx = int(round(pt['time'] * 60.0))
            real_pos = np.array([pt['x'], pt['y']])
            if frame_idx < sim_frames:
                sim_pos = sim_history[frame_idx]
            else:
                sim_pos = sim_history[-1] if sim_frames > 0 else start_pos
            residuals.append(np.linalg.norm(sim_pos - real_pos))
    return np.array(residuals)


def main():
    print("Loading Ground Truth Data")
    shot1 = load_shot_data("shot1_data.json")
    shot3 = load_shot_data("shot3_data.json")

    start1, vel_guess_1 = get_base_kinematics(shot1)
    start3, vel_guess_3 = get_base_kinematics(shot3)

    initial_guess = [
        0.2, 0.015, 0.85, 0.2,
        vel_guess_1[0], vel_guess_1[1],
        vel_guess_3[0], vel_guess_3[1]
    ]

    bounds = (
        [0.05, 0.005, 0.60, 0.0, vel_guess_1[0] - 2.0, vel_guess_1[1] - 2.0, vel_guess_3[0] - 2.0,
         vel_guess_3[1] - 2.0],
        [0.40, 0.050, 0.99, 0.5, vel_guess_1[0] + 2.0, vel_guess_1[1] + 2.0, vel_guess_3[0] + 2.0, vel_guess_3[1] + 2.0]
    )

    print("\nStarting Optimization (Tolerances Enabled)...")
    result = least_squares(
        objective_function, x0=initial_guess,
        args=(shot1, shot3, start1, start3), bounds=bounds, verbose=2,
        ftol=1e-5, xtol=1e-5  # Added tolerances to prevent hanging!
    )

    mu_s, mu_r, e_c, mu_c = result.x[0:4]
    opt_v3 = result.x[6:8]  # Extract the perfected starting velocity for shot 3

    print("\n========================================")
    print("      CALIBRATION COMPLETE")
    print("========================================")
    print(f"Sliding Friction (mu_s)   : {mu_s:.4f}")
    print(f"Rolling Friction (mu_r)   : {mu_r:.4f}")
    print(f"Cushion Restitution (e_c) : {e_c:.4f}")
    print(f"Cushion Friction (mu_c)   : {mu_c:.4f}")
    print(f"\nAverage Error per Point   : {np.mean(np.abs(result.fun)) * 100:.2f} cm")

    print("\nLaunching Pygame Visualizer for Shot 3...")
    sim = Simulation(n_obj_balls=0, mu_s=mu_s, mu_r=mu_r, e_c=e_c, mu_c=mu_c, k_n=1e4)

    # Spawn at Shot 3's starting position
    sim.positions[0] = start3
    sim.in_play[0] = True

    # Fire with Shot 3's optimized velocity
    sim.strike_cue_ball(velocity_x=opt_v3[0], velocity_y=opt_v3[1], topspin=0.0, sidespin=0.0)

    sim_history = []

    def record_frame(s):
        sim_history.append(s.positions[0].copy())

    sim.run(framerate=60.0, frame_callback=record_frame)

    renderer = Renderer(sim)
    clock = pygame.time.Clock()
    running, frame_idx = True, 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False

        renderer.screen.fill((40, 40, 40))
        renderer.screen.blit(renderer.table, (0, 0))

        # Draw Shot 3's manual clicks
        for pt in shot3:
            px, py = renderer.world_to_screen((pt['x'], pt['y']))
            pygame.draw.circle(renderer.screen, (255, 0, 0), (int(px), int(py)), 5)

        if len(sim_history) > 1:
            points = [renderer.world_to_screen(p) for p in sim_history]
            pygame.draw.lines(renderer.screen, (0, 255, 0), False, points, 3)

        if frame_idx < len(sim_history):
            sim.positions[0] = sim_history[frame_idx]
            frame_idx += 1
        elif len(sim_history) > 0:
            sim.positions[0] = sim_history[-1]

        renderer.update_cue_ball_rotation(dt=1 / 60.0)
        renderer.draw_balls()
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()