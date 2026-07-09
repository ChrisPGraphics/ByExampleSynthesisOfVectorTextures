import math

import numpy as np
from skimage.segmentation import quickshift

import segmentation.base_segmentation as base_segmentation


class QuickShiftSegmentation(base_segmentation.BaseSegmentation):
    def __init__(
            self, ratio=1.0, kernel_size=5, max_dist=10, sigma=0, convert2lab=True,
            min_area: int = 0, max_area: int = math.inf, silent: bool = False
    ):
        super().__init__(min_area, max_area, silent)
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.max_dist = max_dist
        self.sigma = sigma
        self.convert2lab = convert2lab

    def _segment(self, image: np.ndarray, mask: np.ndarray = None):
        segments = quickshift(
            image, ratio=self.ratio, kernel_size=self.kernel_size, max_dist=self.max_dist, sigma=self.sigma,
            convert2lab=self.convert2lab
        )

        result = []
        for segment in np.unique(segments):
            if segment == 0:
                continue

            result.append(segments == segment)

        return result
