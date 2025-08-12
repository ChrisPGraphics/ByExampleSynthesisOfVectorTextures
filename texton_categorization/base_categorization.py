import abc
import typing

import hierarchy_node


class BaseCategorization(abc.ABC):
    def __init__(self, cluster_count: int = 15):
        super().__init__()
        self.cluster_count = cluster_count

    @abc.abstractmethod
    def categorize(self, polygons: typing.List[hierarchy_node.VectorNode]):
        pass

    def get_algorithm_name(self) -> str:
        return self.__class__.__name__
