import edit_operations.base_edit_operation as base_edit_operation
import hierarchy_node


class SmallTextonRemoval(base_edit_operation.BaseEditOperation):
    def __init__(self, area_threshold: float, edit_foreground: bool = True, edit_background: bool = True):
        super().__init__(edit_foreground, edit_background)
        self.area_threshold = area_threshold

    def parse_textons(self, polygons: hierarchy_node.VectorNode):
        removal = []

        for polygon in polygons.children:
            if polygon.get_area() < self.area_threshold:
                removal.append(polygon)

        for r in removal:
            polygons.children.remove(r)
