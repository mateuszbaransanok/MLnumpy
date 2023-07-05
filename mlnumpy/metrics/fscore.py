import numpy as np

from mlnumpy.abc.metric import Metric


class F1Score(Metric):
    def __init__(
        self,
        num_classes: int | None = None,
        average: str = "macro",
        epsilon: float = 1e-8,
    ) -> None:
        self.num_classes = num_classes or 0
        self.average = average
        self.epsilon = epsilon

    def __call__(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
    ) -> float:
        targets, predictions = self._onehot_to_ordinal(targets, predictions)

        true_positives = np.zeros(self.num_classes)
        false_positives = np.zeros(self.num_classes)
        false_negatives = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            mask = targets == i
            true_positives[i] = np.sum(targets[mask] == predictions[mask])
            false_positives[i] = np.sum(targets[predictions == i] != predictions[predictions == i])
            false_negatives[i] = np.sum(targets[mask] != predictions[mask])

        if self.average == "macro":
            precision = np.mean(true_positives / (true_positives + false_positives + self.epsilon))
            recall = np.mean(true_positives / (true_positives + false_negatives + self.epsilon))
            f1_score = np.mean(2 * precision * recall / (precision + recall + self.epsilon))
            return float(f1_score)

        if self.average == "micro":
            total_true_positives = np.sum(true_positives)
            total_false_positives = np.sum(false_positives)
            total_false_negatives = np.sum(false_negatives)
            precision = total_true_positives / (
                total_true_positives + total_false_positives + self.epsilon
            )
            recall = total_true_positives / (
                total_true_positives + total_false_negatives + self.epsilon
            )
            f1_score = 2 * (precision * recall) / (precision + recall + self.epsilon)
            return float(f1_score)

        raise ValueError(f"Average '{self.average}' is not supported")
