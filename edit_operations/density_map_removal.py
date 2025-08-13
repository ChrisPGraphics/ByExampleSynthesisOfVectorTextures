import numpy as np
import tqdm

import common
import edit_operations.base_edit_operation as base_edit_operation
import hierarchy_node


class DensityMapRemoval(base_edit_operation.BaseEditOperation):
    def __init__(self, map_path: str, edit_foreground: bool = True, edit_background: bool = True):
        super().__init__(edit_foreground, edit_background)
        self.map_path = map_path
        self.map = common.loader.load_image(self.map_path, grayscale=True)

    def parse_textons(self, polygons: hierarchy_node.VectorNode):
        map_height, map_width = self.map.shape[:2]

        removal = []
        for polygon in tqdm.tqdm(polygons.children):
            centroid = np.clip(polygon.get_centroid().astype(int), [0, 0], [map_width - 1, map_height - 1])
            probability = self.map[centroid[1], centroid[0]]

            if np.random.random() > probability:
                removal.append(polygon)

        for polygon in removal:
            polygons.children.remove(polygon)
