from pool_simulation.constants import *
import numpy as np
import pytest
from pool_simulation.physics import Simulation
import time


def assert_near_segment(x: np.array, p1: np.array, p2: np.array, tol=1e-3):
    """
    Asserts that point x is near the line segment defined by p1 and p2.
    Default tolerance within 1mm
    """
    if np.linalg.norm(p1 - p2) > 1e-9:
        t_hat = np.dot(x-p1, p2-p1) / (np.linalg.norm(p2-p1) ** 2)
        t_star = min(max(t_hat, 0), 1)
        # Closest point on the segment to x
        closest_point = p1 + t_star * (p2 - p1)
        dist = np.linalg.norm(closest_point - x)
    else:
        dist = np.linalg.norm(p1 - x)

    # Use an 'if' statement and pytest.fail()
    if dist >= tol:
        pytest.fail(
            f"Point {x} is too far from segment [{p1}, {p2}]."
            f"\nDistance was {dist:.4f}, but tolerance is {tol}."
        )

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


# def calculate_sliding_end(x0, v0, w0):
#     cross_prod = CUE_BALL_RADIUS * np.array(-w0[1], w0[0])
#     x1 = x0 + (2.0 / 49.0) * (np.linalg.norm(v0 + cross_prod) / (MU_S * 9.81)) * (6 * v0 - cross_prod)
#     t1 = (2.0 / 7.0) * (np.linalg.norm(v0 + cross_prod) / (MU_S * 9.81))
#     return x1, t1
#
#
# def calculate_stop(x0, v0, w0):
#     # Stop Sliding
#     cross_prod = CUE_BALL_RADIUS * np.array(-w0[1], w0[0])
#     x1 = x0 + (2.0 / 49.0) * (np.linalg.norm(v0 + cross_prod) / (MU_S * 9.81)) * (6 * v0 - cross_prod)
#     t1 = (2.0 / 7.0) * (np.linalg.norm(v0 + cross_prod) / (MU_S * 9.81))
#     v1 = (5.0 / 7.0) * v0 - (2.0 / 7.0) * cross_prod
#     # Stop Rolling
#     t2 = t1 + (7.0/5.0) * np.linalg.norm(v1)/(MU_R * 9.81)
#     x2 = x1 + (7.0/10.0) * np.linalg.norm(v1)/(MU_R * 9.81) * v1
#     return x2, t2
#
#
# def test_ball_sliding_simple():
#     """
#     Test where a ball stops sliding when hit in a straight line.
#     Ball at (0,0) with initial velocity [1,0] m/s and no initial angular velocity.
#     """
#     dt = 0.001
#
#     sim = Simulation(n_balls=0, dt_max=0.001)
#     sim.reset(positions=np.array([0.0, 0.0]))
#     sim.velocities[0] = np.array([1.0, 0.0])
#     sim.angular[0] = np.array([0.0, 0.0, 0.0])
#
#     x, t = calculate_sliding_end(sim.positions[0], sim.velocities[0], sim.angular[0])
#
#     elapsed = 0
#     prev_pos = 0
#     while elapsed < t:
#         prev_pos = sim.positions[0].copy()
#         sim.time_step_state_only(dt)
#         elapsed += dt
#     assert_near_segment(x, prev_pos, sim.positions[0])
#
#
# def test_ball_stop_simple():
#     """
#     Test where a ball stops when hit in a straight line.
#     Ball at (0,0) with initial velocity [1,0] m/s and no initial angular velocity.
#     """
#     dt = 0.001
#
#     sim = Simulation(n_balls=0, dt_max=0.001)
#     sim.reset(positions=np.array([0.0, 0.0]))
#     sim.velocities[0] = np.array([1.0, 0.0])
#     sim.angular[0] = np.array([0.0, 0.0, 0.0])
#
#     x, t = calculate_stop(sim.positions[0], sim.velocities[0], sim.angular[0])
#
#     elapsed = 0
#     prev_pos = 0
#     while elapsed < t:
#         prev_pos = sim.positions[0].copy()
#         sim.time_step_state_only(dt)
#         elapsed += dt
#     assert_near_segment(x, prev_pos, sim.positions[0])
#
#
# def test_ball_sliding_swerve():
#     """
#     Test where a ball stops sliding in a more complex curved path.
#     Ball at (0,0) with initial velocity [1,0] m/s and initial angular velocity of [30, -120, 0] rad/s
#     """
#     dt = 0.001
#
#     sim = Simulation(n_balls=0, dt_max=0.001)
#     sim.reset(positions=np.array([-0.8, 0.0]))
#     sim.velocities[0] = np.array([1.0, 0.0])
#     sim.angular[0] = np.array([0.0, 0.0, 0.0])
#
#     x, t = calculate_sliding_end(sim.positions[0], sim.velocities[0], sim.angular[0])
#
#     elapsed = 0
#     prev_pos = 0
#     while elapsed < t:
#         prev_pos = sim.positions[0].copy()
#         sim.time_step_state_only(dt)
#         elapsed += dt
#     assert_near_segment(x, prev_pos, sim.positions[0])
#
#
# def test_ball_stop_swerve():
#     """
#     Test where a ball stops sliding in a more complex curved path.
#     Ball at (0,0) with initial velocity [1,0] m/s and initial angular velocity of [30, -120, 0] rad/s
#     """
#     dt = 0.001
#
#     sim = Simulation(n_balls=0, dt_max=0.001)
#     sim.reset(positions=np.array([-0.8, 0.0]))
#     sim.velocities[0] = np.array([1.0, 0.0])
#     sim.angular[0] = np.array([0.0, 0.0, 0.0])
#
#     x, t = calculate_stop(sim.positions[0], sim.velocities[0], sim.angular[0])
#
#     elapsed = 0
#     prev_pos = 0
#     while elapsed < t:
#         prev_pos = sim.positions[0].copy()
#         sim.time_step_state_only(dt)
#         elapsed += dt
#     assert_near_segment(x, prev_pos, sim.positions[0])
#
