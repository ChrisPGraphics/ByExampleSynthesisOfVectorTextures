import argparse
import math
import os
import typing

import synthesis


class SynthesisConfig:
    placement_tries: int = 20
    improvement_steps: int = 5
    max_fails: int = math.inf
    selection_probability_decay: float = 2
    log_steps_path: typing.Union[str, None] = None
    radius_percentile: int = 50
    skip_density_correction: bool = False
    default_weights: bool = False
    width: int = 500
    height: int = 500
    intermediate_directory: str
    output_directory: str
    no_layers: bool = False
    no_binary: bool = False

    _description = "Synthesizes a novel texture after 'analyze_texture.py' has completed the preprocessing"

    @staticmethod
    def _add_args(parser):
        pass

    @staticmethod
    def _set_properties(config, args):
        pass

    @classmethod
    def from_argv(cls) -> typing.Self:
        parser = argparse.ArgumentParser(
            description=cls._description,
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
            '--intermediate_directory',
            help="The path to the intermediate directory to store converted files"
        )
        parser.add_argument(
            '--output_directory',
            help="The path to the output directory to store synthetic files including the result"
        )
        parser.add_argument(
            '--log_level', default='INFO', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
            help="The logging level to run the script as"
        )
        parser.add_argument(
            '--width', default=500, type=int, help="The width of the resulting image, default is 500"
        )
        parser.add_argument(
            '--height', default=500, type=int, help="The height of the resulting image, default is 500"
        )
        parser.add_argument('--placement_tries', default=20, type=int)
        parser.add_argument('--improvement_steps', default=5, type=int)
        parser.add_argument('--max_fails', default=math.inf, type=int)
        parser.add_argument('--selection_probability_decay', default=2, type=float)
        parser.add_argument('--log_steps_path', default=None, type=str)
        parser.add_argument('--radius_percentile', default=50, type=int)
        parser.add_argument(
            '--no_layers', action='store_true',
            help="Do not save each layer individually"
        )
        parser.add_argument(
            '--no_binary', action='store_true',
            help="Do not save binary files for further editing"
        )
        parser.add_argument(
            '--skip_density_correction', action='store_true',
            help="If enabled, there will be no attempt to correct the per-category density of the primary texton layer"
        )
        parser.add_argument(
            '--default_weights', action='store_true',
            help="Use the default weights instead of fine-tuned weights from the optimizer (if available)"
        )
        parser.add_argument(
            '--weights_file',
            help="The path to a weights file from the optimizer"
        )
        cls._add_args(parser)

        args = parser.parse_args()

        if args.intermediate_directory is None:
            intermediate_directory = os.path.join("intermediate", args.image_name)
        else:
            intermediate_directory = os.path.join(args.intermediate_directory, args.image_name)

        if args.output_directory is None:
            output_directory = os.path.join("output", args.image_name)
        else:
            output_directory = os.path.join(args.output_directory, args.image_name)

        config = cls()

        config.placement_tries = args.placement_tries
        config.improvement_steps = args.improvement_steps
        config.max_fails = args.max_fails
        config.selection_probability_decay = args.selection_probability_decay
        config.log_steps_path = args.log_steps_path
        config.radius_percentile = args.radius_percentile
        config.skip_density_correction = args.skip_density_correction
        config.width = args.width
        config.height = args.height
        config.no_layers = args.no_layers
        config.no_binary = args.no_binary
        config.intermediate_directory = intermediate_directory
        config.output_directory = output_directory

        default_weights_path = os.path.join(intermediate_directory, "placement_weights.json")
        if args.default_weights:
            config.weights = synthesis.Weights()

        elif args.weights_file is not None:
            config.weights, _ = synthesis.Weights.from_json(args.weights_file)

        elif os.path.isfile(default_weights_path):
            config.weights, _ = synthesis.Weights.from_json(default_weights_path)

        else:
            config.weights = synthesis.Weights()

        cls._set_properties(config, args)

        return config
