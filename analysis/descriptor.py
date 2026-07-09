import typing

import numpy as np
import copy


class Descriptor:
    def __init__(self, descriptor: np.ndarray, center: typing.Tuple[int, int]):
        self.descriptor = descriptor
        self.center = center

    def copy(self, deep_copy: bool = True) -> 'typing.Self':
        if deep_copy:
            return copy.deepcopy(self)

        else:
            return copy.copy(self)
