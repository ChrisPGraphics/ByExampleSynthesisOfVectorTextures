import json
import os.path
import typing

import numpy as np

import common


class Weights(common.SavableObject):
    def __init__(
            self,
            empty_area_weight: float = -0.2,
            missed_area_weight: float = -0.4,
            mismatched_area_weight: float = -0.5,
            target_area_weight: float = 0.5,
            same_overlap_area_weight: float = -0.5,
            different_overlap_area_weight: float = -0.5,
    ):
        self.empty_area_weight = empty_area_weight
        self.missed_area_weight = missed_area_weight
        self.mismatched_area_weight = mismatched_area_weight
        self.target_area_weight = target_area_weight
        self.same_overlap_area_weight = same_overlap_area_weight
        self.different_overlap_area_weight = different_overlap_area_weight

    def to_array(self) -> np.ndarray:
        return np.array([
            self.empty_area_weight,
            self.missed_area_weight,
            self.mismatched_area_weight,
            self.target_area_weight,
            self.same_overlap_area_weight,
            self.different_overlap_area_weight
        ])

    @classmethod
    def from_array(cls, array) -> 'typing.Self':
        return cls(*array)

    def to_json(self, filename, score: float, force_overwrite: bool = False):
        if not force_overwrite and os.path.isfile(filename):
            _, current_metadata = self.from_json(filename)
            current_score = current_metadata["score"]

            if score >= current_score:
                return

        result = {
            "weights": self.__dict__,
            "metadata": {
                "score": score
            }
        }

        with open(filename, 'w') as f:
            f.write(json.dumps(result, sort_keys=True, indent=4))

    @staticmethod
    def from_json(filename: str) -> typing.Tuple['Weights', dict]:
        with open(filename, 'r') as f:
            data = json.loads(f.read())

        weights = Weights(**data["weights"])
        metadata = data["metadata"]

        return weights, metadata
