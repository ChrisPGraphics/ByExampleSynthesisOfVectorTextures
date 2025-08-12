import abc
import typing

import numpy as np

import common.loader


class SourceMap(abc.ABC):
    map_count = 0

    @abc.abstractmethod
    def get_distribution(self, position: tuple) -> np.ndarray:
        pass


class EmptySourceMap(SourceMap):
    def __init__(self, default_distro: np.ndarray = None):
        if default_distro is None:
            default_distro = np.ones(1)

        self.default_distro = default_distro

    def get_distribution(self, position: tuple) -> np.ndarray:
        return self.default_distro


class MultiSourceMap(SourceMap):
    def __init__(self, source_map_paths: typing.List[str]):
        raw_source_maps = np.array([
            path if isinstance(path, np.ndarray) else common.loader.load_image(path, grayscale=True)
            for path in source_map_paths
        ])

        raw_source_maps = raw_source_maps / np.sum(raw_source_maps, axis=0, keepdims=True)
        raw_source_maps[np.isnan(raw_source_maps)] = 1 / len(source_map_paths)

        self.map_count = len(source_map_paths)
        self.source_map = np.transpose(raw_source_maps, (1, 2, 0))

    def get_distribution(self, position: tuple) -> np.ndarray:
        return self.source_map[position[1], position[0]]


class BinarySourceMap(MultiSourceMap):
    def __init__(self, source_map_path: str):
        source_map = common.loader.load_image(source_map_path, grayscale=True)

        super().__init__([source_map, 1 - source_map])


class StochasticMap(SourceMap):
    def __init__(self, map_count: int):
        super().__init__()
        self.map_count = map_count
        self.result = np.full(map_count, 1 / map_count)

    def get_distribution(self, position: tuple) -> np.ndarray:
        return self.result
