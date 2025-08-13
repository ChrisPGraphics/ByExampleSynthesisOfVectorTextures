import numpy as np

import edit_operations.base_edit_operation as base_edit_operation
import hierarchy_node


class ProbabilityRemoval(base_edit_operation.BaseEditOperation):
    def __init__(self, probability: float, edit_foreground: bool = True, edit_background: bool = False):
        super().__init__(edit_foreground, edit_background)
        self.probability = probability

    def parse_textons(self, textons: hierarchy_node.VectorNode):
        keep = int(len(textons.children) * self.probability)
        all_indices = np.arange(len(textons.children))
        deletion = list(np.random.choice(all_indices, len(textons.children) - keep, replace=False))
        deletion.sort(reverse=True)

        for i in deletion:
            del textons.children[i]
