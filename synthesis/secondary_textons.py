import logging
import typing

import bridson
import numpy as np
import tqdm

import hierarchy_node
import synthesis.source_map as source_map_models


def secondary_texton_distro(
        secondary_textons: hierarchy_node.VectorNode, shape: tuple, distances: np.ndarray,
        percentile: int = 50, source_map: source_map_models.SourceMap = None
) -> hierarchy_node.VectorNode:

    source_dict = {}
    if source_map is not None:
        source_dict = {i: [] for i in range(source_map.map_count)}

        for child in secondary_textons.children:
            source_dict[child.source_id].append(child)

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
        if source_map is not None:
            source_id = np.random.choice(
                source_map.map_count, p=source_map.get_distribution((int(point[0]), int(point[1])))
            )
            real_choices = source_dict[source_id]
            polygon_index = np.random.randint(0, len(real_choices))
            texton = real_choices[polygon_index].copy(deep_copy=False)

        else:
            polygon_index = np.random.randint(0, choices)
            texton: hierarchy_node.VectorNode = secondary_textons.children[polygon_index].copy(deep_copy=False)

        texton.set_centroid(point)
        result.add_child(texton)

    return result


def secondary_color_adjustment(
        secondary_textons: hierarchy_node.VectorNode, gradient_field: np.ndarray, shape: typing.Tuple[int, int]
):
    logging.info("Recomputing secondary texton color...")
    for texton in secondary_textons.children:
        if texton.color_delta is None:
            continue

        centroid = np.clip(texton.get_centroid().astype(int), 0, shape)
        field_color = gradient_field[*centroid[::-1]]

        color_change = (field_color + texton.color_delta) - texton.color

        for node in texton.level_order_traversal(include_self=True):
            node.original_color = node.color
            node.color = np.clip(node.color + color_change, 0, 1)
