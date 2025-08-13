import math

import numpy as np
import shapely

import edit_operations.base_edit_operation as base_edit_operation
import hierarchy_node


class AddShadow(base_edit_operation.IndependentTextonEditOperation):
    def __init__(
            self, x_angle: float, y_angle: float, intensity: float, element_height: float, edit_foreground: bool = True,
            edit_background: bool = False
    ):
        super().__init__(edit_foreground, edit_background)
        self.x_angle = x_angle
        self.y_angle = y_angle
        self.intensity = intensity
        self.element_height = element_height

        self.x_length = self.element_height / math.tan(self.x_angle)
        self.y_length = self.element_height / math.tan(self.y_angle)

    def edit_texton(self, i: int, texton: hierarchy_node.VectorNode):
        exterior = texton.as_shapely()

        shadow = shapely.affinity.translate(exterior, self.x_length, self.y_length)

        for other in texton.children:
            shadow = shadow.difference(other.as_shapely())

        if isinstance(shadow, shapely.MultiPolygon):
            elements = shadow.geoms

        elif isinstance(shadow, shapely.GeometryCollection):
            elements = [e for e in shadow.geoms if isinstance(e, shapely.Polygon)]

        else:
            elements = [shadow]

        for element in elements:
            shadow_child = hierarchy_node.VectorNode([], color=np.array([0, 0, 0, self.intensity]))
            shadow_child.from_shapely(element)

            if len(shadow_child.exterior) < 2:
                continue

            texton.add_child(shadow_child)
