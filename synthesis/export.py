import os

import PIL.Image
import numpy as np

import hierarchy_node
import analysis


def export_result(
        primary_textons: hierarchy_node.VectorNode, secondary_textons: hierarchy_node.VectorNode,
        gradient_field: analysis.result_objects.GradientFieldResult, gradient_field_raster: np.ndarray,
        output_directory: str, no_binary: bool = False, no_layers: bool = False
):
    os.makedirs(output_directory, exist_ok=True)

    if not no_binary:
        primary_textons.save(os.path.join(output_directory, "primary_textons.dat"))
        secondary_textons.save(os.path.join(output_directory, "secondary_textons.dat"))
        gradient_field.save(os.path.join(output_directory, "gradient_field.dat"))

    primary_raster = primary_textons.to_pil([0, 0, 0, 0])
    secondary_raster = secondary_textons.to_pil([0, 0, 0, 0])
    gradient_field_raster = PIL.Image.fromarray((gradient_field_raster * 255).astype(np.uint8))

    if not no_layers:
        primary_raster.save(os.path.join(output_directory, "primary_textons.png"))
        secondary_raster.save(os.path.join(output_directory, "secondary_textons.png"))
        gradient_field_raster.save(os.path.join(output_directory, "gradient_field.png"))

    gradient_field_raster.paste(secondary_raster, (0, 0), secondary_raster)
    gradient_field_raster.paste(primary_raster, (0, 0), primary_raster)

    gradient_field_raster.save(os.path.join(output_directory, "result.png"))
