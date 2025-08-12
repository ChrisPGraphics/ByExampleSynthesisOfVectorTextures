import numpy as np
import scipy.ndimage as ndimage
import skimage.measure as sk_measure

import common
import hierarchy_node.base_node as base_node


class MaskNode(base_node.BaseNode):
    def __init__(self, mask: np.ndarray):
        super().__init__()
        self.mask = mask

    @classmethod
    def from_image(cls, image: np.ndarray):
        return cls(np.ones(image.shape[:2], dtype=bool))

    def to_polygon(self, pad: int = 1) -> np.ndarray:
        labeled = sk_measure.label(self.mask, return_num=False)

        largest_area = 0
        largest_label = None
        for label in np.unique(labeled):
            if label == 0:
                continue

            current_area = np.count_nonzero(labeled == label)
            if current_area > largest_area:
                largest_area = current_area
                largest_label = label

        polygon = common.binary_operations.mask_to_polygon(labeled == largest_label, pad)
        return np.array([(i[0] - pad, i[1] - pad) for i in polygon])

    def get_area(self) -> int:
        return self.mask.sum()

    def fill_holes(self) -> np.ndarray:
        return ndimage.binary_fill_holes(self.mask.astype(bool))

    def union(self, mask: np.ndarray) -> np.ndarray:
        return np.logical_or(self.mask, mask)

    def intersection(self, mask: np.ndarray) -> np.ndarray:
        return np.logical_and(self.mask, mask)

    def difference(self, mask: np.ndarray) -> np.ndarray:
        return np.logical_and(self.mask, np.logical_not(mask))

    def dilate(self, iterations: int = 1, mask: np.ndarray = None) -> np.ndarray:
        return common.binary_operations.dilate(self.mask, iterations=iterations, mask=mask)

    def is_touching(self, mask: np.ndarray) -> bool:
        return np.any(np.logical_and(self.dilate(1), mask))

    def approximate_centroid(self) -> np.ndarray:
        ys, xs = np.nonzero(self.mask)
        return np.array([xs.mean(), ys.mean()])
