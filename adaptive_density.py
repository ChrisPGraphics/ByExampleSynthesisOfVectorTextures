import os.path
import random
import typing

import argparse
import bridson
import numpy as np
from scipy.spatial import cKDTree

import analysis
import common
import hierarchy_node
import synthesis


def adaptive_primary_textons(textons: hierarchy_node.VectorNode, texton_library: analysis.result_objects.PrimaryTextonResult, position_multiplier: float):
    textons.exterior *= position_multiplier

    initial_textons = [texton.copy(deep_copy=True) for texton in textons.children]
    new_size = (int(textons.get_bounding_width()), int(textons.get_bounding_width()))

    for child in initial_textons:
        child.set_centroid(child.get_centroid() * position_multiplier)

    result = synthesis.primary_texton_distro(
        texton_library.primary_textons, new_size, initial_polygons=initial_textons
    )
    synthesis.global_density_cleanup(
        result, texton_library.global_coverage, texton_library.per_category_coverage, exempt_polygons=initial_textons
    )

    return result


def poisson_infill(points: np.ndarray, spacing: float, size: typing.Tuple[int, int], infill_multiplier=1) -> np.ndarray:
    poisson = np.array(bridson.poisson_disc_samples(*size, spacing))

    tree = cKDTree(points)
    distances, _ = tree.query(poisson, distance_upper_bound=spacing * infill_multiplier)

    return poisson[np.isinf(distances)]


def adaptive_secondary_textons(
        textons: hierarchy_node.VectorNode, texton_library: analysis.result_objects.SecondaryTextonResult, position_multiplier: float
) -> hierarchy_node.VectorNode:

    textons.exterior *= position_multiplier
    new_size = (int(textons.get_bounding_width()), int(textons.get_bounding_width()))

    for child in textons.children:
        child.set_centroid(child.get_centroid() * position_multiplier)

    points = textons.get_child_centroids() * position_multiplier
    new_points = poisson_infill(points, np.percentile(texton_library.element_spacing, 50), new_size)

    for point in new_points:
        texton = random.choice(texton_library.secondary_textons.children).copy(deep_copy=True)
        texton.set_centroid(point)
        textons.add_child(texton)

    return textons


def adaptive_gradient_field(
        synthetic_gradient_field: analysis.result_objects.GradientFieldResult,
        exemplar_gradient_field: analysis.result_objects.GradientFieldResult,
        position_multiplier: float, size

) -> typing.Tuple[analysis.result_objects.GradientFieldResult, np.ndarray]:

    new_size = (size[0] * position_multiplier, size[1] * position_multiplier)
    new_points = poisson_infill(
        synthetic_gradient_field.points * position_multiplier,
        synthetic_gradient_field.query_point_spacing,
        new_size
    )

    synthetic_gradient_field.points = list(synthetic_gradient_field.points * position_multiplier)
    synthetic_gradient_field.colors = list(synthetic_gradient_field.colors)

    for point in new_points:
        synthetic_gradient_field.points.append(point)
        synthetic_gradient_field.colors.append(random.choice(exemplar_gradient_field.colors))

    synthetic_gradient_field.points = np.array(synthetic_gradient_field.points)
    synthetic_gradient_field.colors = np.array(synthetic_gradient_field.colors)

    raster_gradient_field = common.gradient_field.rasterize_rbf(synthetic_gradient_field.points, synthetic_gradient_field.colors, new_size)

    return synthetic_gradient_field, raster_gradient_field


if __name__ == '__main__':
    common.logger.configure_logger()

    parser = argparse.ArgumentParser(
        description="Increases the spacing between textons of a result and then fills in the gaps with new textons to match the exemplar density.",
        epilog='If there are any issues or questions, feel free to visit our GitHub repository at '
               'https://github.com/ChrisPGraphics/ByExampleSynthesisOfVectorTextures'
    )

    parser.add_argument(
        'image_name',
        help="The name of the processed file. "
             "Note that this is not the path to the image nor the directory in the intermediate folder. "
             "Just the base name of the image without file extension. "
             "For example, if the image path is /foo/bar.png and you analyzed the image with "
             "'python analyze_texture.py /foo/bar.png', "
             "you would call this script with 'python synthesize_texture.py bar'"
    )

    parser.add_argument(
        '--multiplier', default=2, type=float,
        help="The multiplier to move textons by. For example, a value of 2 doubles the space between textons."
    )

    args = parser.parse_args()

    analysis_path = os.path.join("intermediate", args.exemplar)
    result_path = os.path.join("output", args.exemplar)

    primary_textons = hierarchy_node.VectorNode.load(os.path.join(result_path, "primary_textons.dat"))
    initial_primary_textons = analysis.result_objects.PrimaryTextonResult.load(os.path.join(analysis_path, "primary_textons.dat"))
    initial_size = primary_textons.get_bounding_size()

    primary_textons = adaptive_primary_textons(primary_textons, initial_primary_textons, args.multiplier)

    secondary_textons = hierarchy_node.VectorNode.load(os.path.join(result_path, "secondary_textons.dat"))
    initial_secondary_textons = analysis.result_objects.SecondaryTextonResult.load(os.path.join(analysis_path, "secondary_textons.dat"))

    secondary_textons = adaptive_secondary_textons(secondary_textons, initial_secondary_textons, args.multiplier)

    gradient_field = analysis.result_objects.GradientFieldResult.load(os.path.join(result_path, "gradient_field.dat"))
    initial_gradient_field = analysis.result_objects.GradientFieldResult.load(os.path.join(analysis_path, "gradient_field.dat"))

    gradient_field, gradient_field_raster = adaptive_gradient_field(gradient_field, initial_gradient_field, args.multiplier, initial_size)

    synthesis.export_result(
        primary_textons, secondary_textons, gradient_field, gradient_field_raster, os.path.join("output", "adaptive_density")
    )
