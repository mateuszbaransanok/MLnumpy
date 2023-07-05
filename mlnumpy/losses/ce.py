import numpy as np

from mlnumpy.abc.activation import Activation
from mlnumpy.abc.loss import Loss
from mlnumpy.activations.sigmoid import Sigmoid
from mlnumpy.activations.softmax import Softmax


class CrossEntropy(Loss):
    def __init__(
        self,
        epsilon: float = 1e-10,
    ) -> None:
        self.epsilon = epsilon

    def __call__(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        error = (targets - predictions) / (
            predictions * (1 - predictions) * targets.shape[0] + self.epsilon
        )
        predictions = np.clip(predictions, self.epsilon, 1 - self.epsilon)
        loss = float(
            np.mean(-(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions)))
        )
        return error, loss

    def validate(
        self,
        activation: Activation,
    ) -> bool:
        return isinstance(activation, (Sigmoid, Softmax))
