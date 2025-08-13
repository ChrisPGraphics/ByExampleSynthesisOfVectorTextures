import argparse
import math
import os
import typing

import synthesis


class InterpolateConfig:
    initial_texture: str
    final_texture: str
    frames: int
    output_directory: str
    save_path: str
    save_layers: bool = False
    no_primary: bool = False
    no_secondary: bool = False
    no_gradient_field: bool = False

    @classmethod
    def from_argv(cls) -> typing.Self:
        parser = argparse.ArgumentParser(
            description="Interpolates between two synthesized textures",
            epilog='If there are any issues or questions, feel free to visit our GitHub repository at '
                   'https://github.com/ChrisPGraphics/ByExampleSynthesisOfVectorTextures'
        )

        parser.add_argument(
            'initial_texture',
            help="The name of the initial texture to interpolate between. "
                 "Note that this is not the path to the image nor the directory in the intermediate folder. "
                 "Just the base name of the image without file extension. "
                 "For example, if the image path is /foo/bar.png and you analyzed the image with "
                 "'python analyze_texture.py /foo/bar.png', "
                 "you would call this script with 'python synthesize_texture.py bar'"
        )
        parser.add_argument(
            'final_texture',
            help="The name of the final texture to interpolate between. See 'initial_texture' for how to enter this field"
        )
        parser.add_argument(
            'frames',
            help="The number of frames to interpolate over (excludes the initial and final frame)",
            type=int
        )
        parser.add_argument(
            '--output_directory', default="output",
            help="The path to the directory where synthesized textures are saved"
        )
        parser.add_argument(
            'save_path',
            help="The directory to save each frame in interpolation"
        )
        parser.add_argument(
            '--log_level', default='INFO', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
            help="The logging level to run the script as"
        )

        parser.add_argument(
            '--save_layers', action='store_true',
            help="Save each layer individually in addition to the final result"
        )
        parser.add_argument(
            '--no_primary', action='store_true',
            help="Do not interpolate the primary texton layer"
        )
        parser.add_argument(
            '--no_secondary', action='store_true',
            help="Do not interpolate the secondary texton layer"
        )
        parser.add_argument(
            '--no_gradient_field', action='store_true',
            help="Do not interpolate the background gradient field"
        )

        args = parser.parse_args()

        config = cls()
        config.initial_texture = args.initial_texture
        config.final_texture = args.final_texture
        config.frames = args.frames
        config.output_directory = args.output_directory
        config.save_path = args.save_path
        config.save_layers = args.save_layers
        config.no_primary = args.no_primary
        config.no_secondary = args.no_secondary
        config.no_gradient_field = args.no_gradient_field

        return config
