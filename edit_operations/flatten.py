import edit_operations.base_edit_operation as base_edit_operation
import hierarchy_node


class Flatten(base_edit_operation.IndependentTextonEditOperation):
    def edit_texton(self, i: int, texton: hierarchy_node.VectorNode):
        texton.children.clear()
