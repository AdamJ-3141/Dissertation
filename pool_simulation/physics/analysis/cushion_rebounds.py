import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pool_simulation.physics import Simulation
from pool_simulation.constants import TABLE_HEIGHT


def execute_shot_trace(sim, speed, top, side, angle_rad, max_x=2.0):
    """Helper function to execute a shot and return a trajectory trace."""
    sim.reset()
    sim.set_up_randomly(0)
    sim.positions[0] = np.array([0.0, 0.0])

    # Calculate initial velocity vector
    vx = np.sin(angle_rad) * speed
    vy = np.cos(angle_rad) * speed

    # Use the force=True override to bypass cue-stick collision validation
    sim.strike_cue_ball(
        vx, vy,
        topspin_offset=top,
        sidespin_offset=side,
        elevation_deg=0.0,
        force=True
    )

    trace = []

    def track_position(s):
        # Stop recording when the ball passes our X limit to keep the plot focused
        if s.positions[0][0] <= max_x:
            trace.append(s.positions[0].copy())

    sim.run(framerate=60, frame_callback=track_position)
    return np.array(trace) if len(trace) > 0 else None


def generate_dissertation_subplots():
    h = TABLE_HEIGHT

    # Infinite horizontal cushions[cite: 2]
    custom_cushions = [
        ((-10.0, h / 2), (10.0, h / 2)),  # Top cushion
        ((-10.0, -h / 2), (10.0, -h / 2))  # Bottom cushion
    ]

    sim = Simulation(
        cushion_line_segments=custom_cushions,
        cushion_circles=[],
        pockets=[]
    )

    angle_rad = np.radians(20)  # 20 degrees from the Y-axis normal
    max_x = 2.0

    # Set up a 1x3 subplot grid
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    cushion_y_top = h / 2 - sim.radii[0]
    cushion_y_bot = -h / 2 + sim.radii[0]

    # SUBPLOT 1: Dense Speed Sweep
    speeds = np.linspace(1, 5.0, 5)  # 8 steps from 0.5 to 4.0 m/s
    colors_speed = cm.Blues(np.linspace(0.4, 1.0, len(speeds)))

    for speed, color in zip(speeds, colors_speed):
        trace = execute_shot_trace(sim, speed, 0.4, 0.0, angle_rad, max_x)
        if trace is not None:
            axs[0].plot(trace[:, 0], trace[:, 1], color=color, linewidth=1.5, label=f"{speed:.1f} m/s")

    axs[0].set_title("Varying Speed (No Spin)")

    # SUBPLOT 2: Dense Topspin/Backspin Sweep
    spins_tb = np.linspace(-0.75, 0.75, 7)  # 9 steps from heavy backspin to heavy topspin
    colors_tb = cm.coolwarm(np.linspace(0.0, 1.0, len(spins_tb)))  # Blue -> Red

    for spin, color in zip(spins_tb, colors_tb):
        # Hold speed constant at 2.0 m/s
        trace = execute_shot_trace(sim, 3.0, spin, 0.0, angle_rad, max_x)
        if trace is not None:
            label = f"{spin:.2f}" + (" (Top)" if spin > 0 else " (Back)" if spin < 0 else " (Stun)")
            axs[1].plot(trace[:, 0], trace[:, 1], color=color, linewidth=1.5, label=label)

    axs[1].set_title("Varying Top/Backspin (Constant 3 m/s)")

    # SUBPLOT 3: Dense Sidespin Sweep
    spins_side = np.linspace(-0.75, 0.75, 7)  # 9 steps from Left to Right spin
    colors_side = cm.PRGn(np.linspace(0.1, 0.9, len(spins_side)))  # Purple (Left) -> Green (Right)

    for spin, color in zip(spins_side, colors_side):
        # Hold speed constant at 2.0 m/s
        trace = execute_shot_trace(sim, 3.0, 0.0, spin, angle_rad, max_x)
        if trace is not None:
            label = f"{spin:.2f}" + (" (Right)" if spin > 0 else " (Left)" if spin < 0 else " (Center)")
            axs[2].plot(trace[:, 0], trace[:, 1], color=color, linewidth=1.5, label=label)

    axs[2].set_title("Varying Sidespin (Constant 3 m/s)")

    # Global Formatting
    for ax in axs:
        ax.axhline(cushion_y_top, color='black', linewidth=4)
        ax.axhline(cushion_y_bot, color='black', linewidth=4)
        ax.set_xlim(-0.1, max_x)
        ax.set_ylim(-h / 2 - 0.05, h / 2 + 0.05)
        ax.set_xlabel("Table X (m)")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right', fontsize='large')

    axs[0].set_ylabel("Table Y (m)")

    plt.suptitle("Stronge-Compliant Cushion Dynamics (20° Incidence)", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig("cushion_rebounds_subplots.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    generate_dissertation_subplots()