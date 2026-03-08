import numpy as np

class Agent:
    def __init__(self):
        pass
    def get_shot_parameters(self, colours, in_play, positions) -> np.ndarray:
        pass
    def get_cue_ball_in_hand_position(self, colours, in_play, positions) -> np.ndarray:
        pass

class Human(Agent):
    def __init__(self):
        super().__init__()
    def get_cue_ball_in_hand_position(self, colours, in_play, positions) -> np.ndarray:
        pass
    def get_shot_parameters(self, colours, in_play, positions) -> np.ndarray:
        pass