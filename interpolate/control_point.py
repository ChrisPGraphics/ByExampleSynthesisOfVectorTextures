import numpy as np


class ControlPoint:
    def __init__(self, position, color):
        self.position: np.ndarray = position
        self.color: np.ndarray = color
