import numpy as np

import common
import hierarchy_node


class PrimaryTextonResult(common.SavableObject):
    def __init__(
            self, primary_textons: hierarchy_node.VectorNode, descriptor_size: int,
            global_coverage: float, per_category_coverage: dict
    ):
        self.primary_textons: hierarchy_node.VectorNode = primary_textons
        self.descriptor_size = descriptor_size
        self.global_coverage = global_coverage
        self.per_category_coverage = per_category_coverage


class SecondaryTextonResult(common.SavableObject):
    def __init__(self, secondary_textons: hierarchy_node.VectorNode, element_spacing: np.ndarray):
        self.secondary_textons = secondary_textons
        self.element_spacing = element_spacing


class GradientFieldResult(common.SavableObject):
    def __init__(self, points: np.ndarray, colors: np.ndarray, query_point_spacing: float, solid_color: np.ndarray):
        self.points = points
        self.colors = colors
        self.query_point_spacing = query_point_spacing
        self.solid_color = solid_color
