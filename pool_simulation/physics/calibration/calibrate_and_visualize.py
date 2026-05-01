import json
import pygame
import numpy as np
from scipy.optimize import least_squares
from pool_simulation.physics.engine_testing import Simulation
from pool_simulation.render.pygame_renderer import Renderer
import matplotlib.pyplot as plt
import cv2


def generate_first_frame_background(video_path):
    print(f"\nExtracting background frame from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video.")
        return None

    # Grab the very first frame
    ret, frame = cap.read()
    cap.release()
    if not ret: return None

    h, w = frame.shape[:2]

    # --- Lens Calibration ---
    focal_length = w * (63 / 100.0)
    cx, cy = (w / 2) + (((64 - 100) / 100.0) * (w * 0.2)), (h / 2) + (((0 - 100) / 100.0) * (h * 0.2))
    cam_matrix = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([(113 - 100) / 500.0, (77 - 100) / 500.0, 0, 0, 0], dtype=np.float32)

    # --- Homography Mapping ---
    VISUAL_W, VISUAL_H = 1000, 500
    tripod_pts = np.array([[230, 98], [1753, 111], [987, 936], [-9, 914]], dtype=np.float32)
    visual_dst_pts = np.array([[0, 0], [VISUAL_W, 0], [VISUAL_W / 2.0, VISUAL_H], [0, VISUAL_H]], dtype=np.float32)
    visual_matrix = cv2.getPerspectiveTransform(tripod_pts, visual_dst_pts)

    # Flatten and stretch the single frame
    frame = cv2.undistort(frame, cam_matrix, dist_coeffs)
    bg_frame = cv2.warpPerspective(frame, visual_matrix, (VISUAL_W, VISUAL_H))

    # Convert BGR (OpenCV) to RGB (Matplotlib)
    return cv2.cvtColor(bg_frame, cv2.COLOR_BGR2RGB)

def generate_dissertation_plot(real_data, sim_history, bg_image=None, ind=None):
    """
    Generates a scatter plot showing the ground truth data points (dots)
    overlaid with the engine's simulated trajectory (solid lines).
    """
    plt.figure(figsize=(10, 5))
    w, h = 1.8288 / 2, 0.9144 / 2

    # Use distinct, high-contrast colors suitable for academic printing
    colors = {
        0: 'black',  # Cue Ball
        1: 'gold',  # Object Ball 1
        2: 'red',  # Object Ball 2
        3: 'blue',
        4: 'green'
    }

    if bg_image is not None:
        # Extent forces the image corners to perfectly match physics coordinates
        plt.imshow(bg_image, extent=(-w, w, -h, h), origin='upper', alpha=0.7)

    # Plot the Ground Truth Data (Scatter Points)
    for bid, pts in real_data.items():
        color = colors.get(bid % len(colors), 'gray')
        label = 'Cue Ball (Real)' if bid == 0 else f'Ball {bid} (Real)'

        real_x = [pt['x'] for pt in pts]
        real_y = [pt['y'] for pt in pts]

        plt.scatter(real_x, real_y, c=color, label=label, alpha=0.6, s=30, edgecolors='none')
        plt.plot(real_x, real_y, c=color, alpha=0.5, linestyle='--', linewidth=1)


    # Plot the Simulated Trajectory (Solid Lines)
    if len(sim_history) > 0:
        bids = sim_history[0].keys()
        for bid in bids:
            color = colors.get(bid % len(colors), 'gray')
            label = 'Cue Ball (Sim)' if bid == 0 else f'Ball {bid} (Sim)'

            sim_x = [frame[bid][0] for frame in sim_history]
            sim_y = [frame[bid][1] for frame in sim_history]

            plt.plot(sim_x, sim_y, c=color, label=label, linewidth=2, alpha=0.8)

    # Draw the Table Bounds (Cushions)
    w, h = 1.8288 / 2, 0.9144 / 2
    plt.plot([-w, w, w, -w, -w], [h, h, -h, -h, h], 'k--', linewidth=1.5, label='Cushion Edge')

    # Formatting for a clean, academic look
    plt.title('Least-Squares Optimization: Ground Truth vs. Simulation', fontsize=14)
    plt.xlabel('X Position (meters)', fontsize=12)
    plt.ylabel('Y Position (meters)', fontsize=12)
    plt.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1.25, 1))

    plt.axis('equal')  # Forces the plot to be drawn to accurate physical scale
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()

    # Save a high-res version directly to folder
    plt.savefig(f"calibration_result_plot{ind}.png", dpi=300, bbox_inches='tight')

    # Display the window
    plt.show()


def load_shot_data(filename):
    with open(filename, 'r') as f:
        raw_data = json.load(f)

    # Find the global starting time to normalize
    t0 = min(d['time'] for d in raw_data)

    data_by_ball = {}
    for d in raw_data:
        # Default to 0 for backwards compatibility with old json files
        bid = d.get('ball_id', 0)
        if bid not in data_by_ball:
            data_by_ball[bid] = []
        data_by_ball[bid].append({'time': d['time'] - t0, 'x': d['x'], 'y': d['y']})

    return data_by_ball


def get_base_kinematics(data_by_ball, frames_to_average=4):
    start_positions = {}
    for bid, pts in data_by_ball.items():
        start_positions[bid] = np.array([pts[0]['x'], pts[0]['y']])

    cb_pts = data_by_ball.get(0, [])
    n = min(len(cb_pts), frames_to_average)

    if n > 1:
        # Extract the timestamps and coordinates
        t_vals = np.array([pt['time'] for pt in cb_pts[:n]])
        x_vals = np.array([pt['x'] for pt in cb_pts[:n]])
        y_vals = np.array([pt['y'] for pt in cb_pts[:n]])

        # Find the line of best fit for position over time (the slope IS the velocity)
        vx_fit, _ = np.polyfit(t_vals, x_vals, 1)
        vy_fit, _ = np.polyfit(t_vals, y_vals, 1)

        # Calculate base speed and normalize the vector to get the pure angle
        speed_guess = np.hypot(vx_fit, vy_fit)

        if speed_guess > 0:
            dir_vector = np.array([vx_fit / speed_guess, vy_fit / speed_guess])
        else:
            dir_vector = np.array([0.0, 0.0])
    else:
        speed_guess = 0.0
        dir_vector = np.array([0.0, 0.0])

    return start_positions, dir_vector, speed_guess

def simulate_trajectory(physics_params, start_positions, cue_velocity, spin):
    mu_s, mu_r, e_c, mu_c, e_b, mu_b = physics_params

    max_id = max(start_positions.keys()) if start_positions else 0
    sim = Simulation(n_obj_balls=max_id, mu_s=mu_s, mu_r=mu_r, e_c=e_c, mu_c=mu_c, restitution=e_b, mu_b=mu_b, k_n=1e4)

    for bid, pos in start_positions.items():
        sim.positions[bid] = pos
        sim.in_play[bid] = True

    # NEW: Pass the optimized spin offsets into the strike!
    sim.strike_cue_ball(
        velocity_x=cue_velocity[0],
        velocity_y=cue_velocity[1],
        topspin_offset=spin[0],
        sidespin_offset=spin[1],
        force=True
    )

    history = []

    def record_frame(s):
        frame_data = {}
        for bid in start_positions.keys():
            current_pos = s.positions[bid].copy()

            # Catch the ball if the engine sent it to the shadow realm
            if current_pos[0] == 999.0:
                # Freeze it at the exact pocket coordinate it fell into
                if len(history) > 0:
                    current_pos = history[-1][bid].copy()
                else:
                    current_pos = start_positions[bid].copy()

            frame_data[bid] = current_pos
        history.append(frame_data)

    sim.run(framerate=60.0, frame_callback=record_frame)
    return history


def objective_function(params, target_shot, shot_data, start_positions, dir_vector, decay_rate=1.5):
    physics_params = params[0:6]

    # Map the new 12-parameter array
    if target_shot == 1:
        speed, spin = params[6], params[7:9]
    else:
        speed, spin = params[9], params[10:12]

    # Reconstruct the 2D velocity from the fixed camera angle
    vel = dir_vector * speed

    residuals = []
    sim_history = simulate_trajectory(physics_params, start_positions, vel, spin)
    sim_frames = len(sim_history)

    for bid, pts in shot_data.items():
        for pt in pts:
            frame_idx = int(round(pt['time'] * 60.0))
            real_pos = np.array([pt['x'], pt['y']])

            if frame_idx < sim_frames:
                sim_pos = sim_history[frame_idx][bid]
            else:
                sim_pos = sim_history[-1][bid] if sim_frames > 0 else start_positions[bid]

            raw_error = np.linalg.norm(sim_pos - real_pos)
            weight = np.exp(-decay_rate * pt['time'])
            residuals.append(raw_error * weight)

    return np.array(residuals)

def main():
    print("Loading Ground Truth Data")
    shot1 = load_shot_data("shot1_data.json")
    shot3 = load_shot_data("shot3_data.json")

    start1, dir1, speed1_guess = get_base_kinematics(shot1)
    start3, dir3, speed3_guess = get_base_kinematics(shot3)

    # Total parameters drops to 12
    initial_guess = [
        0.2, 0.015, 0.85, 0.2, 0.98, 0.01,
        speed1_guess, 0.0, 0.0,
        speed3_guess, -0.4, 0.0
    ]

    bounds = (
        [0.05, 0.005, 0.60, 0.0, 0.80, 0.00,
         max(0.1, speed1_guess - 0.5), -0.75, -0.75,
         max(0.1, speed3_guess - 0.5), -0.5, -0.75],

        [0.40, 0.050, 0.99, 0.5, 1.00, 0.10,
         speed1_guess + 0.5, 0.75, 0.75,
         speed3_guess + 0.5, -0.1, 0.75]
    )

    print("\n--- STAGE 1: Calibrating Table Parameters (Shot 1) ---")
    result_1 = least_squares(
        objective_function, x0=initial_guess,
        args=(1, shot1, start1, dir1), bounds=bounds, verbose=2,  # Added dir1
        ftol=1e-6, xtol=1e-6
    )

    opt_mu_s, opt_mu_r, opt_e_c, opt_mu_c = result_1.x[0:4]

    print("\n--- STAGE 2: Calibrating Ball Parameters (Shot 3) ---")
    initial_guess_2 = result_1.x.copy()
    initial_guess_2[10:12] = [-0.4, 0.0]  # Indices shifted down by 2

    lw = 0.005
    bounds_2 = (
        [max(0.05, opt_mu_s - lw), max(0.005, opt_mu_r - lw), max(0.60, opt_e_c - lw), max(0.0, opt_mu_c - lw),
         0.80, 0.00,
         max(0.1, speed1_guess - 0.5), -0.75, -0.75,
         max(0.1, speed3_guess - 0.5), -0.5, -0.75],

        [min(0.40, opt_mu_s + lw), min(0.050, opt_mu_r + lw), min(0.99, opt_e_c + lw), min(0.5, opt_mu_c + lw),
         1.00, 0.10,
         speed1_guess + 0.5, 0.75, 0.75,
         speed3_guess + 0.5, -0.1, 0.75]
    )

    result = least_squares(
        objective_function, x0=initial_guess_2,
        args=(3, shot3, start3, dir3), bounds=bounds_2, verbose=2,  # Added dir3
        ftol=1e-3, xtol=1e-3, max_nfev=100
    )

    # Unpack the 12 results and reconstruct the 2D arrays for the visualizer
    mu_s, mu_r, e_c, mu_c, e_b, mu_b = result.x[0:6]

    opt_v1 = dir1 * result.x[6]
    opt_spin1 = result.x[7:9]

    opt_v3 = dir3 * result.x[9]
    opt_spin3 = result.x[10:12]

    print("\n========================================")
    print("      CALIBRATION COMPLETE")
    print("========================================")
    print(f"Sliding Friction (mu_s)   : {mu_s:.4f}")
    print(f"Rolling Friction (mu_r)   : {mu_r:.4f}")
    print(f"Cushion Restitution (e_c) : {e_c:.4f}")
    print(f"Cushion Friction (mu_c)   : {mu_c:.4f}")
    print(f"Ball Restitution (e_b)    : {e_b:.4f}")
    print(f"Ball Friction (mu_b)      : {mu_b:.4f}")
    print(f"\nShot 3 Opt Velocity       : [{opt_v3[0]:.4f}, {opt_v3[1]:.4f}]")
    print(f"Shot 3 Opt Spin           : Top: {opt_spin3[0]:.4f}, Side: {opt_spin3[1]:.4f}")
    print(f"\nAverage Error per Point   : {np.mean(np.abs(result.fun)) * 100:.2f} cm")

    print("\nSimulating Final Trajectory for Shot 1...")
    max_id_1 = max(start1.keys()) if start1 else 0
    sim1 = Simulation(n_obj_balls=max_id_1, mu_s=mu_s, mu_r=mu_r, e_c=e_c, mu_c=mu_c, restitution=e_b, mu_b=mu_b, k_n=1e4)

    for bid, pos in start1.items():
        sim1.positions[bid] = pos
        sim1.in_play[bid] = True

    sim1.strike_cue_ball(
        velocity_x=opt_v1[0],
        velocity_y=opt_v1[1],
        topspin_offset=opt_spin1[0],
        sidespin_offset=opt_spin1[1],
        force=True
    )

    sim_history1 = []

    def record_frame_1(s):
        frame_data = {}
        for bid in start1.keys():
            current_pos = s.positions[bid].copy()
            if current_pos[0] == 999.0:
                if len(sim_history1) > 0:
                    current_pos = sim_history1[-1][bid].copy()
                else:
                    current_pos = start1[bid].copy()
            frame_data[bid] = current_pos
        sim_history1.append(frame_data)

    sim1.run(framerate=60.0, frame_callback=record_frame_1)

    # Execution will pause here until you close the Shot 1 Matplotlib window!
    bg1 = generate_first_frame_background("Shot_1.mp4")
    generate_dissertation_plot(shot1, sim_history1, ind=1, bg_image=bg1)

    print("\nSimulating Final Trajectory for Shot 3...")
    max_id_3 = max(start3.keys()) if start3 else 0
    sim3 = Simulation(n_obj_balls=max_id_3, mu_s=mu_s, mu_r=mu_r, e_c=e_c, mu_c=mu_c, restitution=e_b, mu_b=mu_b, k_n=1e4)

    for bid, pos in start3.items():
        sim3.positions[bid] = pos
        sim3.in_play[bid] = True

    sim3.strike_cue_ball(
        velocity_x=opt_v3[0],
        velocity_y=opt_v3[1],
        topspin_offset=opt_spin3[0],
        sidespin_offset=opt_spin3[1],
        force=True
    )

    sim_history3 = []

    def record_frame_3(s):
        frame_data = {}
        for bid in start3.keys():
            current_pos = s.positions[bid].copy()
            if current_pos[0] == 999.0:
                if len(sim_history3) > 0:
                    current_pos = sim_history3[-1][bid].copy()
                else:
                    current_pos = start3[bid].copy()
            frame_data[bid] = current_pos
        sim_history3.append(frame_data)

    sim3.run(framerate=60.0, frame_callback=record_frame_3)

    bg3 = generate_first_frame_background("Shot_3.mp4")
    generate_dissertation_plot(shot3, sim_history3, ind=3, bg_image=bg3)

    # Launch Pygame
    renderer = Renderer(sim3)
    clock = pygame.time.Clock()
    running, frame_idx = True, 0

    colors = [(255, 255, 255), (255, 0, 0), (255, 255, 0), (0, 0, 255), (0, 255, 0), (255, 165, 0)]

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False

        renderer.screen.fill((40, 40, 40))
        renderer.screen.blit(renderer.table, (0, 0))

        for bid, pts in shot3.items():
            ui_color = colors[bid % len(colors)]
            for pt in pts:
                px, py = renderer.world_to_screen((pt['x'], pt['y']))
                pygame.draw.circle(renderer.screen, ui_color, (int(px), int(py)), 5, 1)

        if len(sim_history3) > 1:
            for bid in start3.keys():
                ui_color = colors[bid % len(colors)]
                points = [renderer.world_to_screen(frame[bid]) for frame in sim_history3]
                pygame.draw.lines(renderer.screen, ui_color, False, points, 3)

        if frame_idx < len(sim_history3):
            for bid in start3.keys():
                sim3.positions[bid] = sim_history3[frame_idx][bid]
            frame_idx += 1
        elif len(sim_history3) > 0:
            for bid in start3.keys():
                sim3.positions[bid] = sim_history3[-1][bid]

        renderer.update_cue_ball_rotation(dt=1 / 60.0)
        renderer.draw_balls()
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()