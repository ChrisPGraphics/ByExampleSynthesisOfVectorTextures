import logging
import skimage.measure as sk_measure

import numpy as np
import tqdm

import segmentation.base_segmentation as base_segmentation


class SAMSegmentation(base_segmentation.BaseSegmentation):
    def __init__(
            self, checkpoint_key: str = "default", checkpoint_path: str = "sam_checkpoints/sam_vit_h_4b8939.pth",
            min_area: int = 0, max_area: int = None, silent: bool = False, query_point_spacing: float = 5,
            points_per_batch: int = 50
    ):
        import segment_anything

        super().__init__(min_area, max_area, silent=silent)
        self.checkpoint_key = checkpoint_key
        self.checkpoint_path = checkpoint_path
        self.query_point_spacing = query_point_spacing
        self.points_per_batch = points_per_batch

        logging.info("Loading saved SAM checkpoint...")
        self.sam = segment_anything.sam_model_registry[self.checkpoint_key](checkpoint=self.checkpoint_path)

    def _segment(self, image: np.ndarray, mask: np.ndarray = None):
        import segment_anything

        height, width, _ = image.shape
        points_per_side = int(round(min(width, height) * (1 / self.query_point_spacing), 0))

        logging.info(
            "Automatically detecting segments with {} query points per side ({} total)...".format(
                points_per_side, points_per_side ** 2
            )
        )

        mask_generator = segment_anything.SamAutomaticMaskGenerator(
            self.sam, points_per_side=points_per_side, points_per_batch=self.points_per_batch
        )
        detected_masks = mask_generator.generate((image * 255).astype(np.uint8))
        logging.info("Found {} masks".format(len(detected_masks)))

        logging.info("Separating disjoint masks...")
        accepted_masks = []
        for detected in tqdm.tqdm(detected_masks):
            mask = detected["segmentation"]

            labeled_mask, number = sk_measure.label(mask, return_num=True)

            for label in range(1, number + 1):
                accepted_masks.append(labeled_mask == label)

        logging.info("Found {} disjoint masks".format(len(accepted_masks)))

        logging.info("Removing redundant overlapping masks...")
        real_masks = []
        accepted_masks.sort(key=lambda x: np.count_nonzero(x), reverse=True)

        offset = 0
        for mask_a in tqdm.tqdm(accepted_masks):
            if np.any(mask_a[:, :offset + 1]) and np.any(mask_a[:, -(offset + 1):]):
                continue

            if np.any(mask_a[:offset + 1]) and np.any(mask_a[-(offset + 1):]):
                continue

            mask_a_area = np.count_nonzero(mask_a)

            for mask_b in real_masks:
                intersection_area = np.count_nonzero(np.logical_and(mask_a, mask_b))
                if intersection_area / mask_a_area > 0.9:
                    break

            else:
                real_masks.append(mask_a)

        logging.info("Found {} masks".format(len(real_masks)))

        return real_masks
