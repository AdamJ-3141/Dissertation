import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pool_simulation.physics.engine import Simulation


def run_performance_benchmark():
    sim = Simulation(n_obj_balls=15)

    # --- Experiment Parameters ---
    ball_counts = [1, 3, 5, 7, 9, 11, 13, 15]
    velocities = [2.0, 4.0, 6.0, 8.0, 10.0]  # m/s
    trials_per_setup = 50  # Average across 5 random layouts for stability

    results = []

    print("Starting Headless Benchmark...")
    total_runs = len(ball_counts) * len(velocities) * trials_per_setup
    current_run = 0

    for n_balls in ball_counts:
        for v in velocities:
            times = []

            for _ in range(trials_per_setup):
                sim.set_up_randomly(n_balls)

                random_spin = np.random.uniform(-10.0, 10.0)
                sim.strike_cue_ball(velocity_x=v, velocity_y=np.random.uniform(-0.5, 0.5),
                                    topspin=random_spin, sidespin=random_spin)

                start_time = time.perf_counter()
                sim.run()
                end_time = time.perf_counter()

                times.append((end_time - start_time) * 1000)

                current_run += 1
                if current_run % 20 == 0:
                    print(f"Progress: {current_run}/{total_runs} shots simulated...")

            avg_time = np.mean(times)
            results.append({
                "Object Balls": n_balls,
                "Shot Velocity (m/s)": v,
                "Compute Time (ms)": avg_time
            })

    # --- Visualization ---
    df = pd.DataFrame(results)

    # Pivot the data so Balls are rows and Velocities are columns
    pivot_table = df.pivot(index="Object Balls", columns="Shot Velocity (m/s)", values="Compute Time (ms)")

    plt.figure(figsize=(10, 7))
    sns.set_theme(style="whitegrid")

    # Create the heatmap
    ax = sns.heatmap(
        pivot_table,
        annot=True,  # Show the millisecond values in the boxes
        fmt=".1f",  # 1 decimal place
        cmap="YlOrRd",  # Yellow (Fast) to Red (Slow)
        cbar_kws={'label': 'Average Compute Time (ms)'}
    )

    ax.invert_yaxis()  # Put 1 ball at the bottom, 15 balls at the top
    plt.title("Physics Engine Performance:\nCompute Time vs. Table Complexity", fontsize=16, pad=15)
    plt.tight_layout()

    # Save it in high-res
    plt.savefig("dissertation_heatmap.png", dpi=300)
    print("\nBenchmark Complete! Saved 'dissertation_heatmap.png'")
    plt.show()


if __name__ == "__main__":
    run_performance_benchmark()