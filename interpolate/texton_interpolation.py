import mixbox
import numpy as np
from skimage import color

import hierarchy_node
import interpolate.assignment as assignment
import interpolate.polygon_interpolation as polygon_interpolation


class ColorInterpolator:
    def interpolate(self, color_a: np.ndarray, color_b: np.ndarray, t: float) -> np.ndarray:
        pass

    def get_name(self) -> str:
        return self.__class__.__name__


class RGBColorInterpolation(ColorInterpolator):
    def interpolate(self, color_a: np.ndarray, color_b: np.ndarray, t: float) -> np.ndarray:
        return (1 - t) * color_a + t * color_b


class LABColorInterpolation(ColorInterpolator):
    def interpolate(self, color_a: np.ndarray, color_b: np.ndarray, t: float) -> np.ndarray:
        color_a = color.rgb2lab(color_a)
        color_b = color.rgb2lab(color_b)
        result = (1 - t) * color_a + t * color_b

        return color.lab2rgb(result)


class MixBoxColorInterpolation(ColorInterpolator):
    def interpolate(self, color_a: np.ndarray, color_b: np.ndarray, t: float) -> np.ndarray:
        return np.array(mixbox.lerp(color_a * 255, color_b * 255, t), dtype=np.float32) / 255


class TextonInterpolation:
    def __init__(self, texton_a: hierarchy_node.VectorNode, texton_b: hierarchy_node.VectorNode):
        self.texton_a = texton_a.copy(deep_copy=True)
        self.texton_b = texton_b.copy(deep_copy=True)

        self.texton_a.set_exterior(self.texton_a.as_shapely().buffer(0))
        self.texton_b.set_exterior(self.texton_b.as_shapely().buffer(0))

        self.texton_a.set_centroid((0, 0))
        self.texton_b.set_centroid((0, 0))

        self.interpolation = polygon_interpolation.PolygonInterpolation(self.texton_a.exterior, self.texton_b.exterior)
        self.interpolation.initial_color = self.texton_a.color
        self.interpolation.final_color = self.texton_b.color

    def get_interpolation(
            self, t: float, color_t: float = None, set_category: bool = True, set_descriptor: bool = True,
            color_interpolation_space: ColorInterpolator = None
    ) -> hierarchy_node.VectorNode:
        if color_t is None:
            color_t = t

        if color_interpolation_space is None:
            color_interpolation_space = RGBColorInterpolation()

        if len(self.texton_a.children) == 0 and len(self.texton_b.children) == 0:
            child_interpolations = []
        else:
            child_interpolations, _ = assignment.interpolation_assignment(
                self.texton_a.children, self.texton_b.children, silent=True
            )

        interpolated = self.interpolation(t)
        new_texton = hierarchy_node.VectorNode(
            interpolated, color_interpolation_space.interpolate(
                self.interpolation.initial_color, self.interpolation.final_color, color_t
            )
        )

        # new_texton_shapely = new_texton.as_shapely().buffer(0)
        for child in child_interpolations:
            child_interp = child(t)
            new_child = hierarchy_node.VectorNode(
                child_interp, color_interpolation_space.interpolate(child.initial_color, child.final_color, color_t)
            )
            new_child.color = np.clip(new_child.color, 0, 1)

            # intersection = new_child.as_shapely().buffer(0).intersection(new_texton_shapely)
            # if intersection.area == 0:
            #     continue
            #
            # if isinstance(intersection, shapely.Polygon):
            # new_child.set_exterior(intersection)
            new_texton.add_child(new_child)

        if np.random.random() < t:
            inherit_texton = self.texton_a
        else:
            inherit_texton = self.texton_b

        if set_category:
            new_texton.category = inherit_texton.category

        if set_descriptor:
            new_texton.descriptor = inherit_texton.descriptor.copy()

        return new_texton
