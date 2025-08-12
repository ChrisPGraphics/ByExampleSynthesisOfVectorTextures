import logging
import math
import os
import typing

import PIL.Image
import cv2
import numpy as np
import scipy
import tqdm

import analysis
import common
import hierarchy_node


def analyze_texture(config: analysis.AnalysisConfig) -> typing.Tuple[
    analysis.result_objects.PrimaryTextonResult,
    analysis.result_objects.SecondaryTextonResult,
    analysis.result_objects.GradientFieldResult
]:
    logging.info("Reading image '{}'...".format(os.path.abspath(config.exemplar_path)))
    image = common.loader.load_image(config.exemplar_path)

    logging.info("Extracting primary textons...")
    primary_masks, primary_remainder = analysis.extract_textons(image, config.primary_segmentation)

    coverage_percent = 1 - np.count_nonzero(primary_remainder) / primary_remainder.size
    coverage_pixels = primary_remainder.size - np.count_nonzero(primary_remainder)

    if config.secondary_segmentation is None:
        logging.info("Secondary texton extraction is disabled. Skipping...")
        secondary_masks = []
        secondary_remainder = primary_remainder
        distances = np.array([])

    else:
        logging.info("Extracting secondary textons...")
        secondary_masks, secondary_remainder = analysis.extract_textons(
            image, config.secondary_segmentation, mask=primary_remainder
        )

        if config.secondary_promotion_percentile != 0 and config.secondary_promotion_percentile is not None:
            logging.info("Promiting secondary textons")
            analysis.promote_textons(primary_masks, secondary_masks, config.secondary_promotion_percentile)

        distances = analysis.get_secondary_spacing(np.array([mask.approximate_centroid() for mask in secondary_masks]), primary_remainder)

    logging.info("Converting primary masks into textons...")
    primary_textons = hierarchy_node.VectorNode.from_rectangle(image.shape[:2][::-1])
    for mask in tqdm.tqdm(primary_masks):
        primary_textons.add_child(analysis.mask_to_primary_texton(mask, image, config))

    logging.info("Converting secondary masks into textons...")
    secondary_textons = hierarchy_node.VectorNode.from_rectangle(image.shape[:2][::-1])
    for mask in tqdm.tqdm(secondary_masks):
        secondary_textons.add_child(analysis.mask_to_secondary_texton(mask, image, config))

    try:
        logging.info("Extracting data points for gradient field...")
        gradient_field_colors, gradient_field_points, density = analysis.get_background_gradient_field(
            image, secondary_remainder, density=config.background_query_point_spacing
        )
    except scipy.spatial.QhullError:
        logging.error(
            "Not enough points could be extracted to create a background gradient field! "
            "Using solid colored background instead!"
        )

        gradient_field_colors = []
        gradient_field_points = []
        density = math.inf

    gradient_backup_color = np.median(image[secondary_remainder], axis=0)

    if len(gradient_field_colors) > 0:
        logging.info("Computing color deltas...")
        for polygon in tqdm.tqdm(secondary_textons.children):
            background_color = common.gradient_field.interpolate(
                gradient_field_points, gradient_field_colors, density, polygon.get_centroid()
            )
            polygon.color_delta = polygon.color - background_color

    else:
        logging.warning(
            "Not enough pixels are left over to build a background gradient velocity_field! "
            "Use less aggressive segmentation parameters to fix this problem"
        )

    logging.info("Removing polygons that are too close to the edge...")
    analysis.remove_edge_textons(primary_textons)
    analysis.remove_edge_textons(secondary_textons)

    logging.info("Categorizing polygons using {}...".format(config.primary_texton_clustering.get_algorithm_name()))
    config.primary_texton_clustering.categorize(primary_textons.children)

    logging.info("Computing per category coverage...")
    per_category_coverage = analysis.compute_category_coverage(primary_textons, coverage_pixels, primary_remainder)

    logging.info("Computing primary texton descriptors...")
    descriptor_size = analysis.get_descriptors(
        primary_textons, image.shape[:2][::-1], average_included=config.descriptor_average_included
    )

    logging.info("{} selectable primary textons remain".format(len(primary_textons.children)))

    return (
        analysis.result_objects.PrimaryTextonResult(
            primary_textons, descriptor_size, coverage_percent, per_category_coverage
        ),
        analysis.result_objects.SecondaryTextonResult(
            secondary_textons, distances
        ),
        analysis.result_objects.GradientFieldResult(
            gradient_field_points, gradient_field_colors, config.background_query_point_spacing, gradient_backup_color
        )
    )


def save_result(
        config: analysis.AnalysisConfig,
        primary_textons: analysis.result_objects.PrimaryTextonResult,
        secondary_textons: analysis.result_objects.SecondaryTextonResult,
        gradient_field: analysis.result_objects.GradientFieldResult
):
    logging.info("Saving extraction files to {}...".format(os.path.abspath(config.intermediate_path)))
    os.makedirs(config.intermediate_path, exist_ok=True)

    logging.info("Saving data files...")
    primary_textons.save(os.path.join(config.intermediate_path, "primary_textons.dat"))
    secondary_textons.save(os.path.join(config.intermediate_path, "secondary_textons.dat"))
    gradient_field.save(os.path.join(config.intermediate_path, "gradient_field.dat"))

    logging.info("Converting and copying exemplar...")
    image = common.loader.load_image(config.exemplar_path)
    bgr_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(config.intermediate_path, "exemplar.png"), bgr_image)

    logging.info("Saving raster version of extractions...")
    primary_textons.primary_textons.to_raster(
        os.path.join(config.intermediate_path, "primary_textons.png"),
        background_color=np.array([0, 0, 0, 0])
    )
    secondary_textons.secondary_textons.to_raster(
        os.path.join(config.intermediate_path, "secondary_textons.png"),
        background_color=np.array([0, 0, 0, 0])
    )

    if len(gradient_field.points) >= 2:
        background_gradient = common.gradient_field.rasterize_rbf(
            gradient_field.points, gradient_field.colors, image.shape[:2][::-1]
        )
        bgr_image = cv2.cvtColor((background_gradient * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(config.intermediate_path, "gradient_field.png"), bgr_image)

    else:
        logging.warning("Not enough query points are available for a gradient field! Using solid color instead")
        img_rgb = np.full((*image.shape[:2], 3), gradient_field.solid_color * 255, dtype=np.uint8)
        bgr_image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(config.intermediate_path, "gradient_field.png"), bgr_image)

    raster = PIL.Image.open(os.path.join(config.intermediate_path, "gradient_field.png"))
    secondary_textons = PIL.Image.open(os.path.join(config.intermediate_path, "secondary_textons.png"))
    primary_textons = PIL.Image.open(os.path.join(config.intermediate_path, "primary_textons.png"))

    raster.paste(secondary_textons, (0, 0), secondary_textons)
    raster.paste(primary_textons, (0, 0), primary_textons)
    raster.save(os.path.join(config.intermediate_path, "vector_representation.png"))


if __name__ == '__main__':
    common.logger.configure_logger()
    cfg = analysis.AnalysisConfig.from_argv()

    layers = analyze_texture(cfg)

    save_result(cfg, *layers)
