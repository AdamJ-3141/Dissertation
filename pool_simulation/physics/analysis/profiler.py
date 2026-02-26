import cProfile
import pstats
import numpy as np
from pool_simulation.physics.engine import Simulation


def profile_worst_case_scenario():
    sim = Simulation(n_obj_balls=15)

    # 1. Set up a clustered table
    sim.set_up_randomly(15)

    # 2. Fire the absolute maximum chaos shot
    sim.strike_cue_ball(
        velocity_x=10.0,
        velocity_y=0.5,
        topspin=-10.0,
        sidespin=20.0
    )

    print("Running profiler on a 15-ball, 10m/s shot...")

    # 3. Wrap the run() method in the profiler
    profiler = cProfile.Profile()
    profiler.enable()

    sim.run()

    profiler.disable()

    # 4. Print the top 20 most time-consuming functions
    print("\n--- Profiling Results ---")
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(20)


if __name__ == "__main__":
    profile_worst_case_scenario()