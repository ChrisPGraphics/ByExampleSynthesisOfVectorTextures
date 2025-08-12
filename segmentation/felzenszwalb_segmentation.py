import math

import numpy as np
import skimage
from skimage.segmentation import felzenszwalb

import segmentation.base_segmentation as base_segmentation
import common


class FelzenszwalbSegmentation(base_segmentation.BaseSegmentation):
    def __init__(
            self, scale: float, sigma: float = 0.95, min_area: int = 0,
            max_area: int = math.inf, silent: bool = False
    ):
        super().__init__(0, max_area, silent)

        self.scale = scale
        self.sigma = sigma
        self.min_size = min_area

    def _segment(self, image: np.ndarray, mask: np.ndarray = None):
        cropped_image, cropped_mask, bounding_box = common.binary_operations.crop_to_mask(image, mask)

        segments = felzenszwalb(
            cropped_image, scale=self.scale, sigma=self.sigma, min_size=self.min_size
        ) + 1
        common.binary_operations.apply_mask(segments, cropped_mask)
        segments = common.binary_operations.uncrop_array(segments, mask.shape, bounding_box)

        result = []
        for segment in np.unique(segments):
            if segment == 0:
                continue

            result.append(segments == segment)

        return result
