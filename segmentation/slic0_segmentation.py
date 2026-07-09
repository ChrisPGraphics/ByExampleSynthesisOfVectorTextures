import math

import numpy as np
from skimage.segmentation import slic

import segmentation.base_segmentation as base_segmentation


class SLIC0Segmentation(base_segmentation.BaseSegmentation):
    def __init__(self, segment_count: int = 100, min_area: int = 0, max_area: int = math.inf, silent: bool = False):
        super().__init__(min_area, max_area, silent)
        self.segment_count = segment_count

    def _segment(self, image: np.ndarray, mask: np.ndarray = None):
        segments = slic(image, n_segments=self.segment_count, slic_zero=True)

        result = []
        for segment in np.unique(segments):
            if segment == 0:
                continue

            result.append(segments == segment)

        return result
