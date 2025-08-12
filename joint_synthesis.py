import logging
import os
import typing

import PIL.Image
import numpy as np
from matplotlib import pyplot as plt

import analysis
import common
import synthesis
import hierarchy_node


def synthesize_primary_textons(
        exemplars: typing.List[str], source_map: synthesis.source_map.SourceMap, size: typing.Tuple[int, int]
) -> hierarchy_node.VectorNode:

    source_primaries = [
        analysis.result_objects.PrimaryTextonResult.load("intermediate/{}/primary_textons.dat".format(source)).primary_textons
        for source in exemplars
    ]

    new_parent = hierarchy_node.VectorNode.from_rectangle(
        (
            max([parent.get_bounding_height() for parent in source_primaries]),
            max([parent.get_bounding_width() for parent in source_primaries])
        )
    )

    for source_id, primaries in enumerate(source_primaries):
        for texton in primaries.level_order_traversal():
            texton.source_id = source_id

        new_parent.children.extend(primaries.children)

    return synthesis.primary_texton_distro(new_parent, size, source_map=source_map)


def synthesize_secondary_textons(
        exemplars: typing.List[str], source_map: synthesis.source_map.SourceMap, size: typing.Tuple[int, int]
) -> hierarchy_node.VectorNode:
    source_secondaries = [
        analysis.result_objects.SecondaryTextonResult.load("intermediate/{}/secondary_textons.dat".format(source))
        for source in exemplars
    ]

    distances = []

    new_parent = hierarchy_node.VectorNode.from_rectangle(
        (
            max([parent.secondary_textons.get_bounding_height() for parent in source_secondaries]),
            max([parent.secondary_textons.get_bounding_width() for parent in source_secondaries])
        )
    )

    for source_id, primaries in enumerate(source_secondaries):
        for texton in primaries.secondary_textons.level_order_traversal():
            texton.source_id = source_id

        new_parent.children.extend(primaries.secondary_textons.children)
        distances.extend(primaries.element_spacing)

    distances = np.array(distances)

    return synthesis.secondary_texton_distro(new_parent, size, distances, source_map=source_map)


def synthesize_gradient_field(
        exemplars: typing.List[str], source_map: synthesis.source_map.SourceMap, size: typing.Tuple[int, int]
) -> typing.Tuple[analysis.result_objects.GradientFieldResult, np.ndarray]:

    gradient_data = [
        analysis.result_objects.GradientFieldResult.load("intermediate/{}/gradient_field.dat".format(source))
        for source in exemplars
    ]

    densities = []
    ids = []
    colors = []
    for source_id, primaries in enumerate(gradient_data):
        densities.append(primaries.query_point_spacing)
        colors.extend(primaries.colors)
        ids.extend([source_id] * len(primaries.colors))

    density = np.mean(densities)
    gradient_field_points, gradient_field_colors = synthesis.generate_gradient_field(
        np.array(colors), density, size, source_map, np.array(ids)
    )
    gradient_field_raster = common.gradient_field.rasterize_rbf(
        gradient_field_points, gradient_field_colors, size
    )

    gradient_field = analysis.result_objects.GradientFieldResult(
        gradient_field_points, gradient_field_colors, density, gradient_data[0].solid_color
    )
    return gradient_field, gradient_field_raster


def joint_synthesis(
        config: synthesis.SynthesisConfig,
        exemplars: typing.List[str], source_map: synthesis.source_map.SourceMap, size: typing.Tuple[int, int],
):
    primary_textons = synthesize_primary_textons(exemplars, source_map, size)
    secondary_textons = synthesize_secondary_textons(exemplars, source_map, size)
    gradient_field, gradient_field_raster = synthesize_gradient_field(exemplars, source_map, size)

    synthesis.secondary_color_adjustment(secondary_textons, gradient_field_raster, size)

    logging.info("Saving synthetic files to {}...".format(os.path.abspath(config.output_directory)))
    synthesis.export_result(
        primary_textons, secondary_textons, gradient_field, gradient_field_raster,
        config.output_directory, config.no_binary, config.no_layers
    )


if __name__ == '__main__':
    common.logger.configure_logger()
    cfg = synthesis.SynthesisConfig()
    cfg.output_directory = os.path.join("output", "joint_synthesis")

    joint_synthesis(
        cfg, ["concrete_4", "flowers_1", "tile_1"],
        synthesis.source_map.MultiSourceMap([
            "source_maps/pg_background.png",
            "source_maps/pg_p.png",
            "source_maps/pg_g.png",
        ]),
        (1000, 515)
    )
