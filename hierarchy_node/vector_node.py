import math
import typing

import PIL.Image
import PIL.ImageDraw
import cv2
import numpy as np
from matplotlib import pyplot as plt

import hierarchy_node.base_node as base_node
import shapely
import shapely.affinity
import shapely.coords
import svgwrite
import svgwrite.shapes

if typing.TYPE_CHECKING:
    import analysis

POLYGON_TYPE = typing.Union[np.ndarray, 'VectorNode', shapely.Polygon]


def _convert_to_shapely(polygon) -> shapely.Polygon:
    if isinstance(polygon, VectorNode):
        polygon = polygon.as_shapely()

    elif not isinstance(polygon, shapely.Polygon):
        polygon = shapely.Polygon(polygon)

    return polygon


class VectorNode(base_node.BaseNode):
    def __init__(
            self,
            exterior: typing.Union[np.ndarray, list],
            color: typing.Union[np.ndarray, tuple] = None,
            category: int = None
    ):
        super().__init__()

        if isinstance(exterior, (list, shapely.coords.CoordinateSequence)):
            exterior = np.array(exterior).astype(np.float32)

        self.exterior = exterior
        self.color = color
        self.category = category

        self.color_delta: np.ndarray = None
        self.source_id: int = None
        self.descriptor: 'analysis.Descriptor' = None

        self._cached_shapely = None
        self._cached_shapely_exterior = None

    @classmethod
    def from_rectangle(cls, size: tuple, category: int = None, color: np.ndarray = None) -> typing.Self:
        return cls([(0, 0), (size[0], 0), (size[0], size[1]), (0, size[1]), (0, 0)], category, color)

    @classmethod
    def from_shapely(cls, polygon: shapely.Polygon, category: int = None, color: np.ndarray = None) -> typing.Self:
        exterior = np.array(polygon.exterior.coords).astype(float)
        return cls(exterior, category, color)

    def get_children_by_category(self) -> dict:
        result = {}

        for child in self.children:
            if child.category not in result:
                result[child.category] = []

            result[child.category].append(child)

        return result

    def as_shapely(self) -> shapely.Polygon:
        try:
            if self._cached_shapely is not None:
                if np.array_equal(self._cached_shapely_exterior, self.exterior):
                    return self._cached_shapely

        except AttributeError:
            pass

        p = shapely.Polygon(self.exterior)

        self._cached_shapely_exterior = self.exterior.copy()
        self._cached_shapely = p

        return p

    def set_exterior(self, polygon: shapely.Polygon):
        self.exterior = np.array(polygon.exterior.coords).astype(float)

    def get_centroid(self, yx: bool = False) -> np.ndarray:
        centroid = np.array(self.as_shapely().centroid.coords)[0]

        if yx:
            centroid = centroid[::-1]

        return centroid

    def set_centroid(self, centroid: typing.Union[np.ndarray, tuple]):
        self.move_centroid(centroid - self.get_centroid())

    def move_centroid(self, delta: typing.Union[np.ndarray, tuple]):
        self.exterior = self.exterior.astype(float) + delta

        for child in self.children:
            child.move_centroid(delta)

    def get_child_centroids(self) -> np.ndarray:
        return np.array([i.get_centroid() for i in self.children])

    def get_polygon_coordinate_pairs(self, yx: bool = False, normalized: bool = False) -> np.ndarray:
        coords = self.exterior.copy()

        if normalized:
            coords = self.exterior - self.get_centroid()

        if yx:
            coords = np.flip(coords, axis=1)

        return coords

    def get_polygon_split_coordinates(self, yx: bool = False, normalized: bool = False) -> np.ndarray:
        coords = self.get_polygon_coordinate_pairs(yx, normalized)
        return np.array([coords[:, 0], coords[:, 1]])

    def get_unique_child_categories(self):
        return np.unique([c.category for c in self.children])

    def get_area(self) -> float:
        return self.as_shapely().area

    def get_perimeter(self) -> float:
        return self.as_shapely().length

    def get_polsby_popper_compactness(self) -> float:
        return 4 * math.pi * (self.get_area() / self.get_perimeter() ** 2)

    def get_schwartzberg_compactness(self) -> float:
        return 1 / (self.get_perimeter() / (2 * math.pi * math.sqrt(self.get_area() / math.pi)))

    def get_length_width_ratio(self) -> float:
        return self.get_bounding_height() / self.get_bounding_width()

    def get_reock_score(self) -> float:
        return self.as_shapely().area / self.get_bounding_circle_area()

    def get_convex_hull_score(self) -> float:
        return self.as_shapely().area / self.get_convex_hull().area

    def get_elongation(self) -> float:
        area = self.get_area()
        perimeter = self.get_perimeter()

        return min(area, perimeter) / max(area, perimeter)

    def get_convex_hull(self) -> shapely.Polygon:
        return self.as_shapely().convex_hull

    def get_bounding_box(self) -> np.ndarray:
        min_x, min_y, max_x, max_y = self.as_shapely().bounds

        return np.array([
            (min_x, min_y),
            (min_x, max_y),
            (max_x, max_y),
            (max_x, min_y),
        ])

    def get_bounding_height(self, as_int: bool = False) -> float:
        min_x, min_y, max_x, max_y = self.as_shapely().bounds
        if as_int:
            return int(round(max_x - min_x))
        else:
            return max_x - min_x

    def get_bounding_width(self, as_int: bool = False) -> float:
        min_x, min_y, max_x, max_y = self.as_shapely().bounds
        if as_int:
            return int(round(max_y - min_y))
        else:
            return max_y - min_y

    def get_bounding_size(self, as_int: bool = False) -> typing.Tuple[float, float]:
        min_x, min_y, max_x, max_y = self.as_shapely().bounds
        if as_int:
            return int(round(max_x - min_x)), int(round(max_y - min_y))
        else:
            return max_x - min_x, max_y - min_y

    def distance_to_point(self, point: np.ndarray) -> float:
        return np.linalg.norm(point - self.get_centroid())

    def get_bounding_circle(self) -> shapely.Polygon:
        return shapely.minimum_bounding_circle(self.as_shapely())

    def get_bounding_circle_radius(self) -> float:
        return shapely.minimum_bounding_radius(self.as_shapely())

    def get_bounding_circle_area(self) -> float:
        return math.pi * self.get_bounding_circle_radius() ** 2

    def distance_to_polygon(self, polygon: POLYGON_TYPE) -> float:
        polygon = _convert_to_shapely(polygon)
        return self.distance_to_point(np.array(polygon.centroid.coords)[0])

    def angle_to_point(self, point: np.ndarray) -> float:
        return math.atan2(*(point - self.get_centroid())[::-1])

    def angle_to_polygon(self, polygon: POLYGON_TYPE) -> float:
        polygon = _convert_to_shapely(polygon)
        return self.angle_to_point(np.array(polygon.centroid.coords)[0])

    def get_overlap_percent(self, polygon: POLYGON_TYPE) -> float:
        polygon = _convert_to_shapely(polygon)
        self_polygon = self.as_shapely()

        return polygon.intersection(self_polygon).area / min(polygon.area, self_polygon.area)

    def is_fully_contained(self, polygon: POLYGON_TYPE) -> bool:
        polygon = _convert_to_shapely(polygon)
        self_polygon = self.as_shapely()

        return polygon.intersection(self_polygon).area == polygon.area

    def is_touching(self, polygon: POLYGON_TYPE, border_touching: bool = True) -> bool:
        polygon = _convert_to_shapely(polygon)
        self_polygon = self.as_shapely()

        if border_touching:
            return polygon.intersects(self_polygon)

        return polygon.intersection(self_polygon).area > 0

    def contains_point(self, point) -> bool:
        return self.as_shapely().contains(shapely.Point(point))

    def color_to_int(self, scale: float = 255) -> np.ndarray:
        if self.color is None:
            return None

        return (np.clip(self.color, 0, 1) * scale).astype(int)

    def color_delta_to_int(self, scale: float = 255) -> np.ndarray:
        if self.color_delta is None:
            return None

        return (np.clip(self.color_delta, 0, 1) * scale).astype(int)

    def binary_rasterize(
            self, mask: np.ndarray, centroid: typing.Union[np.ndarray, tuple] = None,
            color: typing.Union[int, None] = 1
    ):
        if color is None:
            color = self.category

        if centroid is None:
            coords = self.exterior

        else:
            coords = self.exterior + (centroid - self.get_centroid())

        points = [coords.astype(int)]

        if mask.dtype == np.uint8:
            cv2.fillPoly(mask, points, int(color))

        else:
            write_mask = np.zeros_like(mask, dtype=np.uint8)
            cv2.fillPoly(write_mask, points, 1)
            mask[write_mask == 1] = int(color)

    def get_raster_coords(
            self, centroid: typing.Union[np.ndarray, tuple] = None, x_lim=None, y_lim=None
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Returns x, y
        :param centroid:
        :param x_lim:
        :param y_lim:
        :return:
        """

        coords = self.exterior

        if centroid is not None:
            coords = coords + (centroid - self.get_centroid())

        lower = coords.min(axis=0)
        upper = coords.max(axis=0)
        size = upper - lower

        mask = np.zeros((size + 1).astype(int), dtype=np.uint8)
        offset_coords = (coords - lower).astype(int)

        cv2.fillPoly(mask, [offset_coords[:, [1, 0]]], 1)
        coordinates = np.where(mask == 1)

        coordinates = np.array([(coordinates[0] + lower[0]).astype(int), (coordinates[1] + lower[1]).astype(int)])

        if x_lim is not None:
            mask = (x_lim[0] < coordinates[0]) & (coordinates[0] < x_lim[1]) & \
                   (y_lim[0] < coordinates[1]) & (coordinates[1] < y_lim[1])

            if not np.any(mask):
                raise OverflowError("Out of range")

            coordinates = (coordinates[0][mask], coordinates[1][mask])

        return coordinates

    def to_svg(self, filename: str, include_self: bool = False, include_children: bool = True):
        drawing = svgwrite.Drawing(filename, profile='tiny', size=self.get_bounding_size(as_int=True))

        if include_self and self.color is not None:
            self._to_svg(drawing)

        if include_children:
            for child in self.level_order_traversal(include_self=False):
                child._to_svg(drawing)

        drawing.save()


    def _to_svg(self, drawing):
        drawing.add(
            svgwrite.shapes.Polygon(
                self.get_polygon_coordinate_pairs(yx=True).tolist(),
                fill=svgwrite.rgb(*np.clip(self.color, 0, 1) * 100, '%')
            )
        )

    def to_raster(self, filename: str, background_color=None, rgba: bool = True, include_self: bool = False, include_children: bool = True):
        self.to_pil(background_color, rgba, include_self, include_children).save(filename)

    def to_pil(self, background_color=None, rgba: bool = True, include_self: bool = False, include_children: bool = True) -> PIL.Image.Image:
        result = PIL.Image.new(
            "RGBA" if rgba else "RGB",
            self.get_bounding_size(True),
            tuple(self.color if background_color is None else background_color)
        )
        result_draw = PIL.ImageDraw.ImageDraw(result, 'RGBA')

        if include_self and self.color is not None:
            self._to_pil(result, result_draw)

        if include_children:
            for child in self.level_order_traversal(include_self=False):
                child._to_pil(result, result_draw)

        return result

    def _to_pil(self, image: PIL.Image.Image, image_draw: PIL.ImageDraw.ImageDraw):
        fill_color = list(self.color_to_int())
        if len(fill_color) == 3:
            fill_color.append(255)

        coordinates = [tuple(c) for c in self.get_polygon_coordinate_pairs().astype(int)]
        if len(coordinates) < 2:
            return

        image_draw.polygon(coordinates, fill=tuple(fill_color))

    def plot(self, include_self: bool = False, include_children: bool = True):
        if include_self and self.color is not None:
            self._plot()

        if include_children:
            for child in self.level_order_traversal(include_self=False):
                child._plot()

    def _plot(self):
        plt.fill(*self.get_polygon_split_coordinates(), c=self.color)

    def to_xml(self):
        pass

    def to_json(self):
        pass
