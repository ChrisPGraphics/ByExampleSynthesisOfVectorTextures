import logging

import numpy as np
import scipy.spatial as spatial


def get_colinear_inter_element_lines(points: np.ndarray) -> list:
    p1, p2 = points[0], points[1]
    line_vector = p2 - p1

    line_unit_vector = line_vector / np.linalg.norm(line_vector)

    projections = np.dot(points - p1, line_unit_vector)

    sorted_indices = np.argsort(projections)
    sorted_points = points[sorted_indices]

    neighbors = [(sorted_points[i], sorted_points[i + 1]) for i in range(len(sorted_points) - 1)]

    return neighbors


def get_inter_element_lines(centroids: np.ndarray, buffer: int = 10) -> np.ndarray:
    min_x, min_y = np.min(centroids, axis=0)
    max_x, max_y = np.max(centroids, axis=0)

    if np.all(np.isclose(centroids[:, 0], centroids[0, 0])) or np.all(np.isclose(centroids[:, 1], centroids[0, 1])):
        logging.warning(
            "All points are colinear! Sorting points along line and using neighbouring points as triangulation"
        )
        triangulation_lines = get_colinear_inter_element_lines(centroids)

    else:
        triangulation = spatial.Delaunay(centroids)

        triangulation_lines = []
        for points in centroids[triangulation.simplices]:
            triangulation_lines.append([points[0], points[1]])
            triangulation_lines.append([points[1], points[2]])
            triangulation_lines.append([points[2], points[0]])

    if len(triangulation_lines) > 5:
        accepted_lines = []
        for line in triangulation_lines:
            start, end = line

            if start[0] <= min_x + buffer or start[0] >= max_x - buffer:
                continue

            if end[0] <= min_x + buffer or end[0] >= max_x - buffer:
                continue

            if start[1] <= min_y + buffer or start[1] >= max_y - buffer:
                continue

            if end[1] <= min_y + buffer or end[1] >= max_y - buffer:
                continue

            accepted_lines.append([start, end])

    else:
        accepted_lines = triangulation_lines

    return np.array(accepted_lines)
