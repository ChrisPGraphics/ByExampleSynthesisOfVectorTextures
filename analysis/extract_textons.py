import datetime
import logging
import typing

import numpy as np
import shapely
import tqdm

import segmentation as segmentation_methods
import hierarchy_node
import analysis


def extract_textons(
        image: np.ndarray, segmentation: segmentation_methods.BaseSegmentation, mask: np.ndarray = None
) -> typing.Tuple[typing.List['hierarchy_node.MaskNode'], np.ndarray]:

    start = datetime.datetime.now()

    if mask is None:
        remaining_mask = np.ones(image.shape[:2]).astype(bool)
    else:
        remaining_mask = mask.copy()

    logging.info("Segmenting textons with {}...".format(segmentation.get_algorithm_name()))
    segments = segmentation.segment(image, mask=remaining_mask)

    logging.info("Computing remaining mask...")
    nodes = []

    for segment in tqdm.tqdm(segments):
        remaining_mask = np.logical_and(remaining_mask, np.logical_not(segment))
        mask_node = hierarchy_node.MaskNode(segment)
        nodes.append(mask_node)

    end = datetime.datetime.now()
    logging.info("Texton extraction took {}".format(end - start))

    return nodes, remaining_mask


def mask_to_primary_texton(mask: hierarchy_node.MaskNode, image: np.ndarray, analysis_config: 'analysis.AnalysisConfig') -> hierarchy_node.VectorNode:
    vector = hierarchy_node.VectorNode(mask.to_polygon(), np.median(image[mask.mask], axis=0))

    if analysis_config.detail_segmentation is not None:
        child_segments = analysis_config.detail_segmentation.segment(image, mask.mask)
        for segment in child_segments:
            vector.add_child(mask_to_detail_texton(hierarchy_node.MaskNode(segment), image, analysis_config))

    return vector


def mask_to_secondary_texton(mask: hierarchy_node.MaskNode, image: np.ndarray, analysis_config: 'analysis.AnalysisConfig') -> hierarchy_node.VectorNode:
    vector = hierarchy_node.VectorNode(mask.to_polygon(), np.median(image[mask.mask], axis=0))

    return vector

def mask_to_detail_texton(mask: hierarchy_node.MaskNode, image: np.ndarray, analysis_config: 'analysis.AnalysisConfig') -> hierarchy_node.VectorNode:
    vector = hierarchy_node.VectorNode(mask.to_polygon(), np.median(image[mask.mask], axis=0))

    return vector


def remove_edge_textons(textons: hierarchy_node.VectorNode, buffer: int = 3):
    min_x, min_y, max_x, max_y = textons.as_shapely().bounds

    for texton in textons.children[:]:
        texton_min_x, texton_min_y = texton.exterior.min(axis=0)
        texton_max_x, texton_max_y = texton.exterior.max(axis=0)

        if (
                texton_min_x < min_x + buffer or
                texton_min_y < min_y + buffer or
                texton_max_x > max_x - buffer or
                texton_max_y > max_y - buffer
        ):
            textons.remove_child(texton)


def compute_category_coverage(primary_textons: hierarchy_node.VectorNode, coverage_pixels, primary_remainder):
    unique_categories = np.unique([c.category for c in primary_textons.children])

    per_category_coverage = {}
    parent_shapely = primary_textons.as_shapely()
    for category in unique_categories:
        polygons = [c.as_shapely().buffer(0) for c in primary_textons.children if c.category == category]
        per_category_coverage[category] = shapely.intersection(parent_shapely, shapely.unary_union(polygons)).area

    scale = coverage_pixels / sum(per_category_coverage.values())
    per_category_coverage = {k: v * scale / primary_remainder.size for k, v in per_category_coverage.items()}

    return per_category_coverage
