import numpy as np

from mlnumpy.abc.optimizer import Optimizer


class AdaDelta(Optimizer):
    def __init__(
        self,
        rho: float = 0.95,
        epsilon: float = 1e-8,
    ) -> None:
        self.rho = rho
        self.epsilon = epsilon
        self.E: dict[int, np.ndarray] = {}
        self.Ed: dict[int, np.ndarray] = {}

    def __call__(
        self,
        weights: np.ndarray,
        gradients: np.ndarray,
    ) -> None:
        i = id(weights)
        self.E[i] = self.rho * self.E.get(i, 0) + (1 - self.rho) * gradients**2
        rms = np.sqrt(self.Ed.get(i, 0) + self.epsilon)

        delta_gradient = rms / np.sqrt(self.E[i] + self.epsilon) * gradients

        self.Ed[i] = self.rho * self.Ed.get(i, 0) + (1 - self.rho) * delta_gradient**2

        weights += delta_gradient
