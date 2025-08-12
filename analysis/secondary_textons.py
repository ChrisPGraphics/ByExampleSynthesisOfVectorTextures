import logging
import typing

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spatial

import common
import hierarchy_node


def promote_textons(
        primary_textons: typing.List[hierarchy_node.MaskNode], secondary_textons: typing.List[hierarchy_node.MaskNode], promote_percentile: float
) -> typing.List[hierarchy_node.MaskNode]:

    area_threshold = np.percentile([i.get_area() for i in primary_textons], promote_percentile)

    logging.info("Promotion area threshold is {}".format(area_threshold))

    promoted = []
    for polygon in secondary_textons:
        if polygon.get_area() >= area_threshold:
            promoted.append(polygon)

    if len(promoted) == 0:
        logging.info("No secondary textons could be promoted")
        return []

    logging.info(
        "Promoted {} of {} secondary textons ({:.2f}%)".format(
            len(promoted), len(secondary_textons), len(promoted) / len(secondary_textons) * 100
        )
    )
    for polygon in promoted:
        primary_textons.append(polygon)
        secondary_textons.remove(polygon)

    return promoted


def get_secondary_spacing(
        centroids: np.ndarray, primary_remainder: np.ndarray = None, buffer: int = 10
) -> np.ndarray:

    if len(centroids) == 0:
        return np.array([])

    try:
        logging.info("Triangulating positions of secondary textons...")
        triangulation_lines = common.geometry_operations.get_inter_element_lines(centroids, buffer)
    except spatial.QhullError:
        return np.array([])

    if primary_remainder is not None:
        logging.info("Removing lines that intersect a primary texton")
        remaining_lines_mask = common.binary_operations.lines_not_touching_mask(triangulation_lines, np.logical_not(primary_remainder))
        triangulation_lines = triangulation_lines[remaining_lines_mask]

    if len(triangulation_lines) == 0:
        return np.array([])

    logging.info("Found {} lines".format(len(triangulation_lines)))

    diffs = triangulation_lines[:, 1, :] - triangulation_lines[:, 0, :]
    distances = np.sqrt((diffs ** 2).sum(axis=1))

    return distances
