import logging

import bridson
import numpy as np
import tqdm

import analysis
import hierarchy_node
import synthesis.synthesis_config as synthesis_config


def secondary_texton_distro(
        secondary_textons: hierarchy_node.VectorNode, shape: tuple, distances: np.ndarray,
        percentile: int = 40
) -> hierarchy_node.VectorNode:

    result = hierarchy_node.VectorNode.from_rectangle(shape)

    if len(distances) == 0:
        logging.warning("No background textons found! Creating solid colored background...")
        return result

    radius = np.percentile(distances, percentile)

    logging.info("Building point distribution...")
    points = bridson.poisson_disc_samples(*shape, radius)

    choices = len(secondary_textons.children)

    logging.info("Placing polygons at points...")
    for point in tqdm.tqdm(points):
        polygon_index = np.random.randint(0, choices)
        texton: hierarchy_node.VectorNode = secondary_textons.children[polygon_index].copy(deep_copy=False)

        texton.set_centroid(point)
        result.add_child(texton)

    return result

