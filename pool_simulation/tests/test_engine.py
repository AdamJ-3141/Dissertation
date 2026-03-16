from pool_simulation.constants import *
import numpy as np
import pytest
from pool_simulation.physics import Simulation
import time

def test_speed():
    # 1. Run a "Dummy" shot to force Numba to compile everything
    sim = Simulation(n_obj_balls=0)
    sim.reset(positions=np.array([[0.0, 0.0]]), colours=np.array([0]), in_play=np.array([True]))
    sim.propel_ball(np.array([True]), np.array([[2.0, 2.0]]), np.array([[0.0, 0.0, 30.0]]))
    sim.run()  # <--- Numba compiles here. It will take ~0.6 seconds.

    # 2. NOW start the real test
    print("Compilation finished. Starting speed test...")
    sim.reset(positions=np.array([[0.0, 0.0]]), colours=np.array([0]), in_play=np.array([True]))
    sim.propel_ball(np.array([True]), np.array([[2.0, 2.0]]), np.array([[0.0, 0.0, 30.0]]))

    start_time = time.perf_counter()
    sim.run()
    end_time = time.perf_counter()

    total_time = end_time - start_time
    print(f"Total time for entire shot (multiple bounces): {total_time:.5f} seconds")


test_speed()


