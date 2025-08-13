import edit_operations.base_edit_operation as base_edit_operation
import hierarchy_node


class TextonReplacement(base_edit_operation.IndependentTextonEditOperation):
    def __init__(self, other_tree: hierarchy_node.VectorNode, edit_foreground: bool = True, edit_background: bool = False):
        super().__init__(edit_foreground, edit_background)
        self.other_tree = other_tree

    def edit_texton(self, i: int, texton: hierarchy_node.VectorNode):
        centroid = texton.get_centroid()
        exterior = texton.as_shapely().buffer(0)

        best_iou = 0
        best_polygon = None

        for other in self.other_tree.children:
            other.set_centroid(centroid)
            other_exterior = other.as_shapely().buffer(0)

            iou = exterior.intersection(other_exterior).area / exterior.union(other_exterior).area

            if iou > best_iou:
                best_iou = iou
                best_polygon = other

        return best_polygon.copy()
