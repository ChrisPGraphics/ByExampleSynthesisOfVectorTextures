import logging
import os
import typing

import PIL.Image
import numpy as np

import analysis
import common
import hierarchy_node
import edit_operations


def edit_texture(output_directory: str, operations: typing.List[edit_operations.BaseEditOperation]):
    logging.info("Reading files into memory...")
    primary_textons = hierarchy_node.VectorNode.load(os.path.join(output_directory, "primary_textons.dat"))
    secondary_textons = hierarchy_node.VectorNode.load(os.path.join(output_directory, "secondary_textons.dat"))
    gradient_field = analysis.result_objects.GradientFieldResult.load(os.path.join(output_directory, "gradient_field.dat"))

    operation_count = len(operations)
    operation_number = 1

    logging.info("Applying edit operations...")
    for operation in operations:

        logging.info("Applying {} ({} of {})".format(operation.get_algorithm_name(), operation_number, operation_count))
        operation.edit_primary_textons(primary_textons)
        operation.edit_secondary_textons(secondary_textons)
        operation.edit_gradient_field(gradient_field)

        operation_number += 1

    logging.info("Rasterizing layers of edited texture...")
    background = common.gradient_field.rasterize_rbf(gradient_field.points, gradient_field.colors, primary_textons.get_bounding_size(as_int=True))
    background = PIL.Image.fromarray((background * 255).astype(np.uint8))

    secondary_textons_raster = secondary_textons.to_pil(rgba=True, background_color=(0, 0, 0, 0))
    primary_textons_raster = primary_textons.to_pil(rgba=True, background_color=(0, 0, 0, 0))

    logging.info("Merging layers to get final result...")
    background.paste(secondary_textons_raster, (0, 0), secondary_textons_raster)
    background.paste(primary_textons_raster, (0, 0), primary_textons_raster)

    logging.info("Saving result to disk...")
    operation_names = [operation.get_algorithm_name() for operation in operations]
    background.save(os.path.join(output_directory, "result_{}.png".format(", ".join(operation_names))))


if __name__ == '__main__':
    common.logger.configure_logger()
    exemplar = "rust_1"

    edits = [
        edit_operations.SmallTextonRemoval(20)
    ]

    edit_texture(os.path.join("output", exemplar), edits)
