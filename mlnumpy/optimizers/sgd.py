import numpy as np

from mlnumpy.abc.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(
        self,
        learning_rate: float = 0.1,
    ) -> None:
        self.learning_rate = learning_rate

    def __call__(
        self,
        weights: np.ndarray,
        gradients: np.ndarray,
    ) -> None:
        weights += self.learning_rate * gradients
