import numpy as np
import shapely
from shapely.geometry import LineString


def polygon_area(poly):
    return 0.5 * np.sum(
        poly[:-1, 0] * poly[1:, 1] - poly[1:, 0] * poly[:-1, 1]
    )


def is_clockwise(poly):
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    return polygon_area(poly) < 0


def insert_random_edge_points(poly: np.ndarray, num_points: int) -> np.ndarray:
    n = len(poly)
    if num_points == 0:
        return poly

    # Treat polygon as closed (connect last point to first)
    closed_poly = np.vstack([poly, poly[0]])
    line = LineString(closed_poly)
    total_length = line.length

    # Generate random distances along the perimeter
    insert_distances = np.sort(np.random.rand(num_points) * total_length)
    inserted = 0
    new_coords = []

    next_dist = insert_distances[inserted] if inserted < num_points else None
    seg_start = 0
    seg_end = 1
    accumulated = 0.0

    while seg_start < len(poly):
        p0 = closed_poly[seg_start]
        p1 = closed_poly[seg_end]
        edge = LineString([p0, p1])
        edge_len = edge.length
        new_coords.append(p0)

        # Insert any points along this edge
        while next_dist is not None and accumulated <= next_dist < accumulated + edge_len:
            local_t = (next_dist - accumulated) / edge_len
            new_point = (1 - local_t) * p0 + local_t * p1
            new_coords.append(new_point)
            inserted += 1
            next_dist = insert_distances[inserted] if inserted < num_points else None

        accumulated += edge_len
        seg_start += 1
        seg_end = (seg_end + 1) % len(closed_poly)

    return np.array(new_coords, dtype=np.float32)


def equalize_polygon_points(poly1: np.ndarray, poly2: np.ndarray):
    len1, len2 = len(poly1), len(poly2)
    if len1 == len2:
        return poly1, poly2
    elif len1 < len2:
        poly1 = insert_random_edge_points(poly1, len2 - len1)
    else:
        poly2 = insert_random_edge_points(poly2, len1 - len2)
    return poly1, poly2


def close_polygon(polygon: np.ndarray) -> np.ndarray:
    if not np.allclose(polygon[0], polygon[-1]):
        polygon = np.vstack([polygon, polygon[0]])

    return polygon


class PolygonInterpolation:
    def __init__(self, polygon_a: np.ndarray, polygon_b: np.ndarray):
        if is_clockwise(polygon_a) != is_clockwise(polygon_b):
            polygon_b = polygon_b[::-1]

        polygon_a, polygon_b = equalize_polygon_points(polygon_a, polygon_b)

        polygon_a = close_polygon(polygon_a)
        polygon_b = close_polygon(polygon_b)

        self.polygon_a = polygon_a
        self.polygon_b = polygon_b

        a_point_count = len(self.polygon_a)
        b_point_count = len(self.polygon_b)

        if a_point_count > b_point_count:
            pass

        self.centroid_a = np.array(shapely.Polygon(self.polygon_a).centroid.xy).flatten()
        self.centroid_b = np.array(shapely.Polygon(self.polygon_b).centroid.xy).flatten()

        self.polygon_a_offset = self.polygon_a - self.centroid_a
        self.polygon_b_offset = self.polygon_b - self.centroid_b

        self.initial_color: np.ndarray = None
        self.final_color: np.ndarray = None

    def interpolate(self, t: float) -> np.ndarray:
        relative_points = (1 - t) * self.polygon_a_offset + t * self.polygon_b_offset
        offset = (1 - t) * self.centroid_a + t * self.centroid_b

        return relative_points + offset

    def __call__(self, t: float) -> np.ndarray:
        return self.interpolate(t)