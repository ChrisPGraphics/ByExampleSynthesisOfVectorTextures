import abc
import base64
import io
import json
import logging
import os.path
import sys
import xml.etree.cElementTree as ET

import PIL.Image
import numpy as np
import svgwrite
import tqdm
from PIL import Image
import argparse

import analysis
import hierarchy_node
import common

EXTENSION_INFERENCE = {
    "xml": "xml",
    "json": "json",
    "svg": "svg",
    "png": "raster",
    "jpg": "raster",
    "jpeg": "raster",
    "bmp": "raster",
    "gif": "raster",
    "webp": "raster",
    "pdf": "raster",
}


class ResultExporter(abc.ABC):
    def __call__(self, output_path: str, result_path: str):
        logging.info("Reading save files at '{}'...".format(os.path.abspath(output_path)))
        primary_textons = hierarchy_node.VectorNode.load(os.path.join(output_path, "primary_textons.dat"))
        secondary_textons = hierarchy_node.VectorNode.load(os.path.join(output_path, "secondary_textons.dat"))
        gradient_field = analysis.result_objects.GradientFieldResult.load(os.path.join(output_path, "gradient_field.dat"))

        if isinstance(primary_textons, analysis.result_objects.PrimaryTextonResult):
            primary_textons = primary_textons.primary_textons

        if isinstance(secondary_textons, analysis.result_objects.SecondaryTextonResult):
            secondary_textons = secondary_textons.secondary_textons

        gradient_field_raster = np.asarray(PIL.Image.open(os.path.join(output_path, "gradient_field.png"))) / 255

        logging.info("Converting files...")
        self._convert_result(primary_textons, secondary_textons, gradient_field, gradient_field_raster, result_path)

    @abc.abstractmethod
    def _convert_result(
            self, primary_textons: hierarchy_node.VectorNode, secondary_textons: hierarchy_node.VectorNode,
            gradient_field: analysis.result_objects.GradientFieldResult, gradient_field_raster: np.ndarray,
            result_path: str
    ):
        pass

    @staticmethod
    def _parent_texton_to_dict(texton: hierarchy_node.VectorNode, string_coordinates: bool) -> dict:
        exterior = " ".join("{},{}".format(*coord) for coord in texton.exterior) if string_coordinates else texton.exterior.astype(float).tolist()
        centroid = "{},{}".format(*texton.get_centroid()) if string_coordinates else texton.get_centroid().astype(float).tolist()

        return dict(
            exterior=exterior,
            category=str(texton.category) if string_coordinates else int(texton.category),
            color='#%02x%02x%02x' % tuple(texton.color_to_int()),
            area=str(texton.get_area()) if string_coordinates else texton.get_area(),
            centroid=centroid
        )

    @staticmethod
    def _child_texton_to_dict(texton: hierarchy_node.VectorNode, string_coordinates: bool) -> dict:
        exterior = " ".join("{},{}".format(*coord) for coord in texton.exterior) if string_coordinates else texton.exterior.astype(float).tolist()
        centroid = "{},{}".format(*texton.get_centroid()) if string_coordinates else texton.get_centroid().astype(float).tolist()

        return dict(
            exterior=exterior,
            color='#%02x%02x%02x' % tuple(texton.color_to_int()),
            color_delta=None if texton.color_delta is None else '#%02x%02x%02x' % tuple(texton.color_delta_to_int()),
            area=str(texton.get_area()) if string_coordinates else texton.get_area(),
            centroid=centroid
        )

    def get_name(self) -> str:
        return self.__class__.__name__


class XMLExporter(ResultExporter):
    def _convert_result(
            self, primary_textons: hierarchy_node.VectorNode, secondary_textons: hierarchy_node.VectorNode,
            gradient_field: analysis.result_objects.GradientFieldResult, gradient_field_raster: np.ndarray,
            result_path: str
    ):
        logging.info("Creating root element...")
        root = ET.Element(
            "result",
            width=str(primary_textons.get_bounding_width(as_int=True)),
            height=str(primary_textons.get_bounding_height(as_int=True)),
            version="1.0.0"
        )

        primary = ET.SubElement(root, "primary", polygon_count=str(len(primary_textons.children)))
        primary_textons.to_xml(primary)

        secondary = ET.SubElement(root, "secondary", polygon_count=str(len(secondary_textons.children)))
        secondary_textons.to_xml(secondary)

        gradient = ET.SubElement(
            root, "gradient", point_count=str(len(gradient_field.points)), point_spacing=str(gradient_field.query_point_spacing)
        )
        for point, color in zip(gradient_field.points, gradient_field.colors):
            ET.SubElement(
                gradient, "point",
                color='#%02x%02x%02x' % tuple((np.clip(color, 0, 1) * 255).astype(int)),
                point="{},{}".format(*point)
            )

        logging.info("Saving to '{}'...".format(os.path.abspath(result_path)))
        tree = ET.ElementTree(root)
        tree.write(result_path, xml_declaration=True, encoding="utf-8")


class JSONExporter(ResultExporter):
    def _convert_result(
            self, primary_textons: hierarchy_node.VectorNode, secondary_textons: hierarchy_node.VectorNode,
            gradient_field: analysis.result_objects.GradientFieldResult, gradient_field_raster: np.ndarray,
            result_path: str
    ):
        result = dict(
            width=primary_textons.get_bounding_width(as_int=True),
            height=primary_textons.get_bounding_height(as_int=True),
            version="1.0.0",
            primary=dict(polygon_count=len(primary_textons.children), textons=primary_textons.to_json(include_self=False)),
            secondary=dict(polygon_count=len(secondary_textons.children), textons=secondary_textons.to_json(include_self=False)),
            gradient=dict(total=len(gradient_field.points), point_spacing=gradient_field.query_point_spacing, points=[]),
        )

        for point, color in zip(gradient_field.points, gradient_field.colors):
            result["gradient"]["points"].append(dict(
                color='#%02x%02x%02x' % tuple((np.clip(color, 0, 1) * 255).astype(int)),
                point=point.astype(float).tolist()
            ))

        with open(result_path, 'w') as f:
            f.write(json.dumps(result, separators=(',', ':')))


class SVGExporter(ResultExporter):
    def _convert_result(
            self, primary_textons: hierarchy_node.VectorNode, secondary_textons: hierarchy_node.VectorNode,
            gradient_field: analysis.result_objects.GradientFieldResult, gradient_field_raster: np.ndarray,
            result_path: str
    ):
        size = (primary_textons.get_bounding_width(), primary_textons.get_bounding_height())

        dwg = svgwrite.Drawing(result_path, profile='tiny', size=size)
        logging.info("Converting gradient velocity_field...")

        image_pil = Image.fromarray((gradient_field_raster * 255).astype(np.uint8))
        png_buffer = io.BytesIO()
        image_pil.save(png_buffer, format='PNG')
        png_data = png_buffer.getvalue()

        base64_png = base64.b64encode(png_data).decode('ascii')
        data_uri = f"data:image/png;base64,{base64_png}"

        dwg.add(dwg.image(href=data_uri, insert=(0, 0), size=size))

        logging.info("Converting secondary textons...")
        secondary_textons.to_svg(dwg, include_self=False)

        logging.info("Converting primary textons...")
        primary_textons.to_svg(dwg, include_self=False)

        logging.info("Saving to '{}'...".format(os.path.abspath(result_path)))
        dwg.save()


class RasterExporter(ResultExporter):
    def _convert_result(
            self, primary_textons: hierarchy_node.VectorNode, secondary_textons: hierarchy_node.VectorNode,
            gradient_field: analysis.result_objects.GradientFieldResult, gradient_field_raster: np.ndarray,
            result_path: str
    ):
        logging.info("Converting gradient velocity_field...")
        background = Image.fromarray((gradient_field_raster * 255).astype(np.uint8))

        logging.info("Converting secondary textons...")
        secondary = secondary_textons.to_pil(background_color=(0, 0, 0, 0), rgba=True)
        background.paste(secondary, (0, 0), secondary)

        logging.info("Converting primary textons...")
        primary = primary_textons.to_pil(background_color=(0, 0, 0, 0), rgba=True)
        background.paste(primary, (0, 0), primary)

        logging.info("Saving to '{}'...".format(os.path.abspath(result_path)))
        background.save(result_path)


if __name__ == '__main__':
    common.logger.configure_logger()

    parser = argparse.ArgumentParser(
        description="Exports a synthesized texture in a wide range of formats after 'synthesize_texture.py'. Note that neither of the following flags can be used when using the synthesis script: '--result_only' '--raster_only'",
        epilog='If there are any issues or questions, feel free to visit our GitHub repository at '
               'https://github.com/ChrisPGraphics/ByExampleSynthesisOfVectorTextures'
    )

    parser.add_argument(
        'input_path',
        help="The path to the directory to convert. "
             "Can either be an analyzed in the 'intermediate' directory or synthetic result in the 'output' directory."
    )

    parser.add_argument(
        'output_path',
        help="The path to the converted output file"
    )

    parser.add_argument(
        '--format', default='infer', choices=['xml', 'json', 'svg', 'raster', 'infer'],
        help="The logging level to run the script as"
    )

    parser.add_argument(
        '--intermediate_directory',
        help="The path to the intermediate directory to store converted files"
    )

    args = parser.parse_args()

    if args.format == "infer":
        logging.info("Inferring desired export format")
        extension = os.path.splitext(args.output_path)[1]
        try:
            args.format = EXTENSION_INFERENCE[extension[1:].lower()]
        except KeyError:
            logging.fatal("Unable to infer export format from file extension '{}'".format(extension))
            sys.exit(1)

    if args.format == "xml":
        exporter = XMLExporter()

    elif args.format == "json":
        exporter = JSONExporter()

    elif args.format == "svg":
        exporter = SVGExporter()

    elif args.format == "raster":
        exporter = RasterExporter()

    else:
        logging.fatal("'{}' is an unrecognized export format".format(args.format))
        sys.exit(1)

    logging.info("Using {}".format(exporter.get_name()))
    exporter(args.input_path, args.output_path)
