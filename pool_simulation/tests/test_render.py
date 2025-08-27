from pool_simulation.render.pygame_renderer import Renderer
from pool_simulation.constants import *
from pool_simulation.physics import Simulation
import numpy as np


def test_coordinate_transform():
    sim = Simulation()
    scale = PIXELS_PER_METER
    renderer = Renderer(sim, scale=scale)

    # Origin (0,0) should map to roughly screen center
    sx, sy = renderer.world_to_screen((0, 0))
    assert np.isclose(sx, renderer.width / 2)
    assert np.isclose(sy, renderer.height / 2)

    # +1 world unit in x should move by about +scale in screen space
    sx2, sy2 = renderer.world_to_screen((1, 0))
    dx, dy = sx2 - sx, sy2 - sy
    assert np.allclose((dx, dy), (scale, 0), atol=1e-6)

    # +1 world unit in y should move by about -scale in screen space
    sx3, sy3 = renderer.world_to_screen((0, 1))
    dx, dy = sx3 - sx, sy3 - sy
    assert np.allclose((dx, dy), (0, -scale), atol=1e-6)

    # Bottom-left table corner should map near top-left of screen
    bl = renderer.world_to_screen((-sim.table_width / 2, -sim.table_height / 2))
    tr = renderer.world_to_screen((sim.table_width / 2, sim.table_height / 2))
    assert bl[0] < tr[0]  # left < right
    assert bl[1] > tr[1]  # bottom is lower on screen
