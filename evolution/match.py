from unittest import case

from pool_simulation.physics import Simulation
from enum import Enum

class Match:
    def __init__(self, engine):
        self.engine: Simulation = engine
        self.turn = 0
        self.open_table = True
        self.turn_state: TurnState = TurnState.BALL_IN_HAND_BAULK

    def play_turn(self):

        match self.turn_state:
            case TurnState.NORMAL:
                pass
            case TurnState.BALL_IN_HAND_BAULK:
                pass
            case TurnState.BALL_IN_HAND:
                pass


class TurnState(Enum):
    NORMAL = 0
    BALL_IN_HAND = 1
    BALL_IN_HAND_BAULK = 2