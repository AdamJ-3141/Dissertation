from pool_simulation.constants import *
import numpy as np
from pool_simulation.physics import Simulation


def test_cueball_movement():
    sim = Simulation(n_balls=0)
