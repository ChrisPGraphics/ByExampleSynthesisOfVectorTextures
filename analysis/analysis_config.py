import argparse
import os
import typing

import segmentation
import texton_categorization


class DefaultValue:
    pass


class AnalysisConfig:
    primary_segmentation: segmentation.BaseSegmentation
    secondary_segmentation: typing.Union[segmentation.BaseSegmentation, None]
    detail_segmentation: typing.Union[segmentation.BaseSegmentation, None]
    exemplar_path: str

    primary_texton_clustering: texton_categorization.BaseCategorization = texton_categorization.ColorAreaCompactnessCategorization()
    descriptor_average_included: float = 2.25
    secondary_promotion_percentile: typing.Union[float, None] = 50
    background_query_point_spacing: float = 25
    intermediate_path: str = "intermediate"

    def __init__(
            self,
            exemplar_path: str,
            primary_segmentation: segmentation.BaseSegmentation = DefaultValue,
            secondary_segmentation: segmentation.BaseSegmentation = DefaultValue,
            detail_segmentation: segmentation.BaseSegmentation = DefaultValue
    ):
        self.exemplar_path = exemplar_path
        self.primary_segmentation = primary_segmentation
        self.secondary_segmentation = secondary_segmentation
        self.detail_segmentation = detail_segmentation

        if self.primary_segmentation == DefaultValue:
            self.primary_segmentation = segmentation.SAMSegmentation(min_area=5)

        if self.secondary_segmentation == DefaultValue:
            self.secondary_segmentation = segmentation.FloodFillSegmentation(0.1, 3, 300)

        if self.detail_segmentation == DefaultValue:
            self.detail_segmentation = segmentation.FelzenszwalbSegmentation(scale=10, min_area=3)

        if self.detail_segmentation is not None:
            self.detail_segmentation.silent = True

    @classmethod
    def from_argv(cls) -> typing.Self:
        parser = argparse.ArgumentParser(
            description="Performs all of the necessary preprocessing before a novel texture can be synthesized",
            epilog='If there are any issues or questions, feel free to visit our GitHub repository at '
                   'https://github.com/ChrisPGraphics/ByExampleSynthesisOfVectorTextures'
        )

        parser.add_argument('input_image', help="The path to the texture that should be processed")
        parser.add_argument(
            '--output_directory',
            help="The path to the intermediate directory to store converted files"
        )
        parser.add_argument(
            '--log_level', default='INFO', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
            help="The logging level to run the script as"
        )
        parser.add_argument(
            '--average_included', type=float, default=2.25,
            help="The average number of textons to be included in each direction of all descriptors"
        )
        parser.add_argument(
            '--promotion_percentile', type=float, default=50,
            help="The size percentile of primary textons that a secondary must fall within to be promoted (or 0 to disable)"
        )
        parser.add_argument(
            '--color_spacing', type=float, default=25,
            help="The spacing between query points when constructing the background color pallet"
        )
        parser.add_argument(
            '--texton_clusters', type=int, default=15,
            help="The number of clusters to group primary textons into"
        )

        args = parser.parse_args()

        if args.output_directory is None:
            output_directory = os.path.join("intermediate", os.path.splitext(os.path.basename(args.input_image))[0])
        else:
            output_directory = args.output_directory

        config_object = cls(args.input_image)

        config_object.descriptor_average_included = args.average_included
        config_object.secondary_promotion_percentile = args.promotion_percentile
        config_object.background_query_point_spacing = args.color_spacing
        config_object.primary_texton_clustering.cluster_count = args.texton_clusters
        config_object.intermediate_path = output_directory

        return config_object
