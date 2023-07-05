import numpy as np

from mlnumpy.abc.optimizer import Optimizer


class AdaGrad(Optimizer):
    def __init__(
        self,
        learning_rate: float = 0.01,
        epsilon: float = 1e-8,
    ) -> None:
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.G: dict[int, np.ndarray] = {}

    def __call__(
        self,
        weights: np.ndarray,
        gradients: np.ndarray,
    ) -> None:
        i = id(weights)
        self.G[i] = self.G.get(i, 0) + gradients**2

        weights += self.learning_rate / np.sqrt(self.G[i] + self.epsilon) * gradients
