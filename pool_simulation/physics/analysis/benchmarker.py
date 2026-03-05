import time
from pool_simulation.physics.engine import Simulation


def benchmark_break_shot(iterations=100):
    sim = Simulation(n_obj_balls=15)

    start_time = time.perf_counter()

    for _ in range(iterations):
        sim.reset_to_break()
        # Hit a massive break
        sim.strike_cue_ball(0.0, 8.5, 5.0, 0.0)

        # Run headlessly (simulates until all balls STOP)
        sim.run()

    end_time = time.perf_counter()
    total_time = end_time - start_time

    print(f"Ran {iterations} break shots in {total_time:.4f} seconds.")
    print(f"Average time per full shot: {(total_time / iterations) * 1000:.2f} ms")


if __name__ == "__main__":
    benchmark_break_shot()