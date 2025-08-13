import abc

import tqdm

import analysis
import hierarchy_node


class BaseEditOperation:
    def __init__(self, edit_primary_textons: bool = True, edit_secondary_textons: bool = False):
        super().__init__()
        self.process_primary_textons = edit_primary_textons
        self.process_secondary_textons = edit_secondary_textons

    @abc.abstractmethod
    def parse_textons(self, textons: hierarchy_node.VectorNode):
        pass

    def edit_primary_textons(self, textons: hierarchy_node.VectorNode):
        if self.process_primary_textons:
            self.parse_textons(textons)

    def edit_secondary_textons(self, textons: hierarchy_node.VectorNode):
        if self.process_secondary_textons:
            self.parse_textons(textons)

    def edit_gradient_field(self, gradient_field: analysis.result_objects.GradientFieldResult):
        pass

    def get_algorithm_name(self) -> str:
        return self.__class__.__name__


class IndependentTextonEditOperation(BaseEditOperation, abc.ABC):
    def parse_textons(self, textons: hierarchy_node.VectorNode):
        for i, texton in tqdm.tqdm(enumerate(textons.children), total=len(textons.children)):
            result = self.edit_texton(i, texton)

            if isinstance(result, hierarchy_node.VectorNode):
                textons.children[i] = result

    @abc.abstractmethod
    def edit_texton(self, i: int, texton: hierarchy_node.VectorNode):
        pass
