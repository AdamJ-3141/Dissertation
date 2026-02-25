import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pool_simulation.physics import Simulation

import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def plot_trajectories(sim, history):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)

    # ==========================================
    # 1. DRAW TABLE GEOMETRY
    # ==========================================
    # Draw line segments
    for p1, p2 in sim.line_segments:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='darkgreen', linewidth=3)

    # Draw circular corners/jaws
    for cx, cy, cr in sim.circles:
        circle = Circle((cx, cy), cr, facecolor='none', edgecolor='darkgreen', linewidth=3)
        ax.add_patch(circle)

    # ==========================================
    # 2. DRAW BALL PATHS
    # ==========================================
    colors = ['white', 'yellow', 'blue', 'red', 'purple', 'orange', 'green', 'maroon', 'black']

    for i, path in history.items():
        if not path or not sim.in_play[i]:
            continue

        path = np.array(path)
        color = colors[sim.colours[i]] if sim.colours[i] < len(colors) else 'gray'
        edge_color = 'black' if color == 'white' else color

        ax.plot(path[:, 0], path[:, 1], color=edge_color, linewidth=2, alpha=0.7, label=f"Ball {i}")

        final_pos = path[-1]
        ball_circle = Circle(final_pos, sim.radii[i], facecolor=color, edgecolor='black', zorder=3)
        ax.add_patch(ball_circle)
        ax.scatter(path[0, 0], path[0, 1], marker='x', color='black', zorder=4)

    plt.title("Compliant Cushion Geometry Test")
    plt.xlabel("Table Length (m)")
    plt.ylabel("Table Width (m)")
    if history:
        plt.legend()
    plt.show()


def borderless_collision():
    # Initialize engine with 1 object ball + cue ball
    sim = Simulation(n_balls=1)

    # Place Object ball at (0, 0), Cue ball at (-0.5, -0.01) to create an offset cut shot
    positions = np.array([
        [-0.5, 0.0],  # Cue Ball
        [1.0, 0.3]  # Object Ball 1
    ])
    colours = np.array([0, 2])  # 0 = White, 2 = Yellow
    in_play = np.array([True, True])

    sim.reset(positions=positions, colours=colours, in_play=in_play)

    # Propel the cue ball with heavy bottom-left spin
    active_mask = np.zeros(sim.n_balls + 1, dtype=bool)
    active_mask[0] = True

    # Velocity: 2.0 m/s to the right. Spin: massive backspin (wy) and left-spin (wz)
    sim.propel_ball(
        ball_mask=active_mask,
        velocities=np.array([[1, 0.0]]),
        angulars=np.array([[-30.0, 50.0, -20.0]])
    )

    history = {i: [] for i in range(sim.n_balls + 1)}

    # 2. Define the callback that run() will trigger
    def tracker(simulation_instance):
        for i in range(simulation_instance.n_balls + 1):
            if simulation_instance.in_play[i]:
                history[i].append(simulation_instance.positions[i].copy())

    # 3. Just call run!
    sim.run(framerate=60.0, frame_callback=tracker, verbose=True)

    # 4. Plot the results
    plot_trajectories(sim, history)


def cushion_collision():
    # Initialize engine with just the cue ball
    sim = Simulation(n_balls=0)

    # Place Cue ball in the center of the table
    positions = np.array([
        [-0.6, -0.3],
    ])
    colours = np.array([0])
    in_play = np.array([True])

    sim.reset(positions=positions, colours=colours, in_play=in_play)

    active_mask = np.zeros(sim.n_balls + 1, dtype=bool)
    active_mask[0] = True

    # Fire the cue ball towards the TOP-RIGHT cushion
    # Give it MASSIVE right-hand english (Z-axis spin)
    sim.propel_ball(
        ball_mask=active_mask,
        velocities=np.array([[1.5, 1]]),
        angulars=np.array([[0.0, 0.0, 40.0]])  # A professional, heavy running spin
    )

    # Set up tracking
    history = {i: [] for i in range(sim.n_balls + 1)}

    def tracker(simulation_instance):
        for i in range(simulation_instance.n_balls + 1):
            if simulation_instance.in_play[i]:
                history[i].append(simulation_instance.positions[i].copy())

    # Run the simulation with verbose logging so you can read the exact Stronge outputs!
    sim.run(framerate=60.0, frame_callback=tracker, verbose=True)

    # Plot the results
    plot_trajectories(sim, history)


if __name__ == "__main__":
    cushion_collision()
