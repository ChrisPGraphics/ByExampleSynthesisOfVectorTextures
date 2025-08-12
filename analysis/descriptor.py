import typing

import numpy as np


class Descriptor:
    def __init__(self, descriptor: np.ndarray, center: typing.Tuple[int, int]):
        self.descriptor = descriptor
        self.center = center
