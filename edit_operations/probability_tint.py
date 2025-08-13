import numpy as np

import edit_operations.base_edit_operation as base_edit_operation
import hierarchy_node


class ProbabilityTint(base_edit_operation.IndependentTextonEditOperation):
    def __init__(
            self, tint_color: tuple, alpha: float, edit_foreground: bool = True, edit_background: bool = False,
            probability: float = 1
    ):
        super().__init__(edit_foreground, edit_background)
        self.alpha = alpha
        self.tint_color = np.array(tint_color)
        self.probability = probability

    def edit_texton(self, i: int, texton: hierarchy_node.VectorNode):
        if np.random.random() > self.probability:
            return

        for child in texton.level_order_traversal():
            child.color = child.color + (self.tint_color - child.color) * self.alpha
