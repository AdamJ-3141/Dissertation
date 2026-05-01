import numpy as np
import matplotlib.pyplot as plt
from pool_simulation.physics import Simulation
from pool_simulation.constants import g


def crude_discrete_trajectory(pos, v, w, sim, dt):
    pos = pos.copy()
    v = v.copy()
    w = w.copy()

    mu_s = sim.mu_s
    mu_r = sim.mu_r
    R = sim.radii[0]
    g = 9.81

    trace = [pos.copy()]
    max_steps = int(100.0 / dt)

    for _ in range(max_steps):
        # Contact velocity: u = v + R * (z-hat x omega)
        u = v + R * np.array([-w[1], w[0]])
        u_norm = np.linalg.norm(u)
        v_norm = np.linalg.norm(v)

        a = np.zeros(2)

        # Calculate the magnitude of change in slip velocity for this dt step
        du_mag = 3.5 * mu_s * g * dt

        # SLIDING phase - but only if the friction won't overshoot u=0 this frame
        if u_norm > 1e-4 and du_mag < u_norm:
            u_hat = u / u_norm
            a = -mu_s * g * u_hat

            # Spin decay: w_dot = (5 / 2R) * mu_s * g * (z-hat x u_hat)[cite: 1]
            w_dot_xy = (5.0 * mu_s * g) / (2.0 * R) * np.array([-u_hat[1], u_hat[0]])
            w[:2] += w_dot_xy * dt

        # ROLLING phase (If u_norm is small, OR if du_mag would have overshot u=0)
        elif v_norm > 1e-4:
            v_hat = v / v_norm
            a = -(5.0 / 7.0) * mu_r * g * v_hat

            # Force pure rolling constraint, which clamps u strictly to 0[cite: 1]
            w[0] = -v[1] / R
            w[1] = v[0] / R
        else:
            break  # STOPPED[cite: 1]

        # Z-axis spin decay[cite: 1]
        decay = (5.0 * sim.mu_sp * g * dt) / (2.0 * R)
        w[2] = np.sign(w[2]) * max(0.0, abs(w[2]) - decay)

        # Standard Euler Step
        pos += v * dt
        v += a * dt

        trace.append(pos.copy())

    return np.array(trace)


def generate_dissertation_plots():
    sim = Simulation(cushion_circles=[], cushion_line_segments=[], pockets=[])

    # Shot parameters (Spin radius = 0.721, safely under the 0.75 miscue limit)
    vx, vy = 1.0, 1.5
    topspin = -0.4
    sidespin = 0.6
    elevation = 60.0

    # Ensure exact static starting position
    start_pos = np.array([0.0, -0.2])

    # Get Analytic Trajectory
    sim.reset()
    sim.set_up_randomly(0)
    sim.positions[0] = start_pos.copy()
    analytic_trace = sim.map_to_first_coll(vx, vy, topspin, sidespin, elevation)
    analytic_trace = np.array(analytic_trace)
    analytic_final_pos = analytic_trace[-1]

    # Capture exact state for Discrete Trajectory
    sim.reset()
    sim.set_up_randomly(0)
    sim.positions[0] = start_pos.copy()
    sim.strike_cue_ball(vx, vy, topspin, sidespin, elevation)

    # Capture the exact state the microsecond after the cue strike
    initial_v = sim.velocities[0].copy()
    initial_w = sim.angular[0].copy()

    dt_visual = 0.1
    discrete_trace = crude_discrete_trajectory(start_pos, initial_v, initial_w, sim, dt_visual)

    # GRAPH 1: Visual Trajectory
    plt.figure(figsize=(8, 6))
    plt.plot(analytic_trace[:, 0], analytic_trace[:, 1], label="Analytic (Event-Based)", color='blue', linewidth=2)
    plt.plot(discrete_trace[:, 0], discrete_trace[:, 1], label=f"Discrete (Euler, dt={dt_visual}s)", color='red',
             linestyle='--')
    plt.scatter([analytic_final_pos[0]], [analytic_final_pos[1]], color='blue', marker='x', s=100)
    plt.scatter([discrete_trace[-1, 0]], [discrete_trace[-1, 1]], color='red', marker='x', s=100)

    plt.title("Trajectory Divergence: Analytic vs Discrete Integration")
    plt.xlabel("Table X (m)")
    plt.ylabel("Table Y (m)")
    plt.legend()
    plt.grid(True)
    plt.savefig("trajectory_comparison.png", dpi=300)
    plt.show()

    # GRAPH 2: Error vs DT
    dt_values = np.logspace(-5, -1, 20)
    errors = []

    for dt in dt_values:
        trace = crude_discrete_trajectory(start_pos, initial_v, initial_w, sim, dt)
        discrete_final_pos = trace[-1]

        # Euclidean distance error between resting positions
        error = np.linalg.norm(discrete_final_pos - analytic_final_pos)
        errors.append(error)

    plt.figure(figsize=(8, 6))
    plt.plot(dt_values, errors, marker='o', color='green')
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Resting Position Error vs Time Step Size ($dt$)")
    plt.xlabel("Time Step $dt$ (seconds) - Log Scale")
    plt.ylabel("Final Position Euclidean Error (m) - Log Scale")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig("dt_error_log.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    generate_dissertation_plots()