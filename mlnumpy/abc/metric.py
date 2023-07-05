from abc import ABC, abstractmethod

import numpy as np


class Metric(ABC):
    @abstractmethod
    def __call__(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
    ) -> float:
        pass

    def setup(
        self,
        num_classes: int,
    ) -> None:
        self.num_classes = num_classes

    @staticmethod
    def _onehot_to_ordinal(
        targets: np.ndarray,
        predictions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(targets.shape) > 1:
            targets = np.argmax(targets, axis=1)
        if len(predictions.shape) > 1:
            predictions = np.argmax(predictions, axis=1)
        return targets, predictions
