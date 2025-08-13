import logging
import os.path

import PIL.Image
import PIL.ImageDraw
import numpy as np
import tqdm

import analysis
import common.logger
import hierarchy_node
import interpolate


def render_polygons(mapping, t: float, size) -> PIL.Image.Image:
    layer = PIL.Image.new("RGBA", size, color=(0, 0, 0, 0))
    draw = PIL.ImageDraw.Draw(layer, 'RGBA')

    for interp in mapping:
        p = interp.interpolate(t)
        color = (1 - t) * interp.initial_color + t * interp.final_color
        coordinates = [tuple(c) for c in p.astype(int)]
        draw.polygon(coordinates, fill=tuple((color * 255).astype(np.uint8)))

    return layer


def interpolate_textures(config: interpolate.InterpolateConfig):
    primary_mapping = None
    secondary_mapping = None
    gradient_point_mapping = None

    source_primaries = hierarchy_node.VectorNode.load(os.path.join(config.output_directory, config.initial_texture, "primary_textons.dat"))
    destination_primaries = hierarchy_node.VectorNode.load(os.path.join(config.output_directory, config.final_texture, "primary_textons.dat"))

    width = int(max(source_primaries.get_bounding_width(), destination_primaries.get_bounding_width()))
    height = int(max(source_primaries.get_bounding_height(), destination_primaries.get_bounding_height()))
    frame_size = (width, height)

    if not config.no_primary:
        logging.info("Computing assignment for primary textons...")
        primary_mapping = interpolate.primary_texton_mapping(source_primaries, destination_primaries)

    if not config.no_secondary:
        logging.info("Computing assignment for secondary textons...")
        source_secondaries = hierarchy_node.VectorNode.load(os.path.join(config.output_directory, config.initial_texture, "secondary_textons.dat"))
        destination_secondaries = hierarchy_node.VectorNode.load(os.path.join(config.output_directory, config.final_texture, "secondary_textons.dat"))

        if len(source_secondaries.children) == 0:
            source_secondaries.children = source_primaries.children

        if len(destination_secondaries.children) == 0:
            destination_secondaries.children = destination_primaries.children

        secondary_mapping, _ = interpolate.assign_interpolation(source_secondaries.children, destination_secondaries.children)

    if not config.no_gradient_field:
        logging.info("Computing assignment for gradient field control points...")
        source_gradient = analysis.result_objects.GradientFieldResult.load(os.path.join(config.output_directory, config.initial_texture, "gradient_field.dat"))
        destination_gradient = analysis.result_objects.GradientFieldResult.load(os.path.join(config.output_directory, config.final_texture, "gradient_field.dat"))
        gradient_point_mapping = interpolate.gradient_point_assignment(source_gradient, destination_gradient)

    logging.info("Rendering result and saving frames to '{}'...".format(os.path.abspath(config.save_path)))
    os.makedirs(config.save_path, exist_ok=True)
    for i, t in tqdm.tqdm(enumerate(np.linspace(0, 1, config.frames + 2)), total=config.frames + 2):
        if gradient_point_mapping is not None:
            colors = []
            positions = []

            for initial_polygon, final_polygons in gradient_point_mapping.items():
                for final_polygon in final_polygons:
                    positions.append((1 - t) * initial_polygon.position + t * final_polygon.position)
                    colors.append((1 - t) * initial_polygon.color + t * final_polygon.color)

            gradient_field = common.gradient_field.rasterize_rbf(np.array(positions), np.array(colors), frame_size)
            frame = PIL.Image.fromarray((gradient_field * 255).astype(np.uint8))

            if config.save_layers:
                frame.save(os.path.join(config.save_path, "{:03d}_gradient_field.png".format(i)))

        else:
            frame = PIL.Image.new("RGBA", frame_size, color=(0, 0, 0, 0))

        if secondary_mapping is not None:
            secondary_layer = render_polygons(secondary_mapping, t, frame_size)
            frame.paste(secondary_layer, (0, 0), secondary_layer)

            if config.save_layers:
                secondary_layer.save(os.path.join(config.save_path, "{:03d}_secondary_textons.png".format(i)))

        if primary_mapping is not None:
            primary_layer = render_polygons(primary_mapping, t, frame_size)
            frame.paste(primary_layer, (0, 0), primary_layer)

            if config.save_layers:
                primary_layer.save(os.path.join(config.save_path, "{:03d}_primary_textons.png".format(i)))

        frame.save(os.path.join(config.save_path, "{:03d}.png".format(i)))


if __name__ == '__main__':
    common.logger.configure_logger()
    cfg = interpolate.InterpolateConfig.from_argv()
    interpolate_textures(cfg)
