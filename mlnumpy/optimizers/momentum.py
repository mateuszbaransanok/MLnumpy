import numpy as np

from mlnumpy.abc.optimizer import Optimizer


class Momentum(Optimizer):
    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
    ) -> None:
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity: dict[int, np.ndarray] = {}

    def __call__(
        self,
        weights: np.ndarray,
        gradients: np.ndarray,
    ) -> None:
        i = id(weights)
        self.velocity[i] = self.momentum * self.velocity.get(i, 0) + self.learning_rate * gradients
        weights += self.velocity[i]
