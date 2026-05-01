import pytest
import numpy as np
from pool_simulation.physics import Simulation
from pool_simulation.constants import *

@pytest.fixture
def engine():
    sim = Simulation()

    sim.positions = np.array([[0.0, 0.0], [0.1, 0.0]])
    sim.radii = np.array([CUE_BALL_RADIUS, OBJECT_BALL_RADIUS])
    sim.in_play = np.array([True, True])

    sim.ball_states = np.array(["STOPPED", "STOPPED"])

    sim.line_segments = [((-0.05, -1.0), (-0.05, 1.0))]
    sim.circles = []

    return sim


def test_open_table_valid(engine):
    # Aiming up (+y). Cushion is left, OB is right.
    # No restrictions, should allow full backspin (-0.8).
    assert engine.validate_shot(velocity_x=0.0, velocity_y=1.0) == True


def test_zero_velocity(engine):
    assert engine.validate_shot(velocity_x=0.0, velocity_y=0.0) == False


def test_cushion_blocking_flat(engine):
    # Cue ball is close to the left cushion (d = 0.05). Aiming right (+x).
    # Shooting perfectly flat (0 elev). Trying to use backspin (-0.5).
    # The shaft will hit the cushion.
    assert engine.validate_shot(velocity_x=1.0, velocity_y=0.0, topspin_offset=-0.5, elevation_deg=0.0) == False


def test_cushion_clearing_elevated(engine):
    # Same setup as above, but jacking up the cue to 45 degrees.
    # The back of the cue is now in the air, clearing the cushion rubber.
    assert engine.validate_shot(velocity_x=1.0, velocity_y=0.0, topspin_offset=-0.5, elevation_deg=45.0) == True


def test_ball_blocking_flat(engine):

    # Aiming left (-x). Cue stick goes backwards (+x) directly through the OB.
    # Trying to hit dead center (topspin = 0.0). Shaft hits the peak of the ball.
    assert engine.validate_shot(velocity_x=-1.0, velocity_y=0.0) == False


def test_ball_clearing_sidespin(engine):

    # This shifts the cue stick laterally and aims slightly to the left so it *just* crosses the "shoulder" of the OB,
    # rather than the high center peak.
    assert engine.validate_shot(velocity_x=-0.99026807, velocity_y=-0.1391731,
                                topspin_offset=0.6, sidespin_offset=0.6,
                                elevation_deg=0.0) == True

def test_ball_clearing_elev(engine):

    # Clearing the top of the cue ball
    assert engine.validate_shot(velocity_x=-1.0, velocity_y=0.0, elevation_deg=19.0) == True