import math

import numpy as np
from skimage.segmentation import watershed

import segmentation.base_segmentation as base_segmentation


class WatershedSegmentation(base_segmentation.BaseSegmentation):
    def __init__(
            self, markers=None, connectivity=1, offset=None, compactness=0, watershed_line=False,
            min_area: int = 0, max_area: int = math.inf, silent: bool = False
    ):
        super().__init__(min_area, max_area, silent)
        self.markers = markers
        self.connectivity = connectivity
        self.offset = offset
        self.compactness = compactness
        self.watershed_line = watershed_line

    def _segment(self, image: np.ndarray, mask: np.ndarray = None):
        segments = watershed(
            image, markers=self.markers, connectivity=self.connectivity, offset=self.offset,
            compactness=self.compactness, watershed_line=self.watershed_line
        )

        segments = segments[:, :, 0]

        result = []
        for segment in np.unique(segments):
            if segment == 0:
                continue

            result.append(segments == segment)

        return result
