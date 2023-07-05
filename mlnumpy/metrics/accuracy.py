import numpy as np

from mlnumpy.abc.metric import Metric


class Accuracy(Metric):
    def __call__(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
    ) -> float:
        targets, predictions = self._onehot_to_ordinal(targets, predictions)
        return float(np.mean(targets == predictions))
