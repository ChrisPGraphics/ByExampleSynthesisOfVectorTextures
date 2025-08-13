import math

import numpy as np
import scipy.spatial as spatial

import edit_operations.base_edit_operation as base_edit_operation
import hierarchy_node


class ForcedAnisotropy(base_edit_operation.IndependentTextonEditOperation):
    def __init__(self, angle: float, edit_foreground: bool = True, edit_background: bool = False):
        super().__init__(edit_foreground, edit_background)
        self.angle = angle

    def edit_texton(self, i: int, texton: hierarchy_node.VectorNode):
        bounds = texton.exterior
        distance_matrix = spatial.distance_matrix(bounds, bounds)

        max_index = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)

        longest_start = bounds[max_index[0]]
        longest_end = bounds[max_index[1]]

        delta = longest_end - longest_start

        estimated_angle = math.atan2(delta[1], delta[0])

        angle_change = self.angle - estimated_angle
        texton.rotate(angle_change)
