import logging
import math
import os
import typing

import PIL.Image
import numpy as np

import analysis
import common
import hierarchy_node
import synthesis


def synthesize_texture(config: synthesis.SynthesisConfig):
    shape = (config.width, config.height)

    logging.info("Reading analyzed files...")
    extracted_primary_textons = analysis.result_objects.PrimaryTextonResult.load(os.path.join(config.intermediate_directory, "primary_textons.dat"))
    extracted_secondary_textons = analysis.result_objects.SecondaryTextonResult.load(os.path.join(config.intermediate_directory, "secondary_textons.dat"))
    extracted_gradient_field = analysis.result_objects.GradientFieldResult.load(os.path.join(config.intermediate_directory, "gradient_field.dat"))

    logging.info("Synthesizing primary texton distribution...")
    primary_textons = synthesis.primary_texton_distro(
        extracted_primary_textons.primary_textons, shape,
        log_steps_directory=config.log_steps_path,
        placement_tries=config.placement_tries,
        improvement_steps=config.placement_tries,
        max_fails=config.max_fails,
        selection_probability_decay=config.selection_probability_decay
    )

    if config.skip_density_correction:
        logging.info("Skipping primary texton density")

    else:
        logging.info("Correcting primary texton density...")
        synthesis.global_density_cleanup(
            primary_textons, extracted_primary_textons.global_coverage, extracted_primary_textons.per_category_coverage,
            exempt_categories=[]
        )

    logging.info("Synthesizing secondary texton distribution...")
    secondary_textons = synthesis.secondary_texton_distro(
        extracted_secondary_textons.secondary_textons, shape, extracted_secondary_textons.element_spacing, config.radius_percentile
    )

    if extracted_gradient_field.query_point_spacing == math.inf or len(extracted_gradient_field.colors) == 0:
        logging.warning("Gradient velocity_field was not successfully extracted! Using solid colored background instead!")
        gradient_field = np.full((*shape[::-1], 3), extracted_gradient_field.solid_color)
        gradient_field_result = extracted_gradient_field

    else:
        logging.info("Synthesizing background gradient field...")
        gradient_field_points, gradient_field_colors = synthesis.generate_gradient_field(
            extracted_gradient_field.colors, extracted_gradient_field.query_point_spacing, shape
        )
        gradient_field = common.gradient_field.rasterize_rbf(gradient_field_points, gradient_field_colors, shape)
        gradient_field_result = analysis.result_objects.GradientFieldResult(
            gradient_field_points, gradient_field_colors, extracted_gradient_field.query_point_spacing,
            extracted_gradient_field.solid_color
        )

    synthesis.secondary_color_adjustment(secondary_textons, gradient_field, shape)

    return primary_textons, secondary_textons, gradient_field_result, gradient_field


def save_result(
        config: synthesis.SynthesisConfig, 
        primary_textons: hierarchy_node.VectorNode, 
        secondary_textons: hierarchy_node.VectorNode, 
        gradient_field_result: analysis.result_objects.GradientFieldResult, 
        gradient_field: np.ndarray
):
    logging.info("Saving synthetic files to {}...".format(os.path.abspath(config.output_directory)))
    synthesis.export_result(
        primary_textons, secondary_textons, gradient_field_result, gradient_field,
        config.output_directory, config.no_binary, config.no_layers
    )


if __name__ == '__main__':
    common.logger.configure_logger()
    cfg = synthesis.SynthesisConfig.from_argv()

    layers = synthesize_texture(cfg)
    save_result(cfg, *layers)
