import numpy as np

from mlnumpy.abc.optimizer import Optimizer


class NesterovMomentum(Optimizer):
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
        previous_velocity = self.velocity.get(i, 0)
        self.velocity[i] = self.momentum * previous_velocity + self.learning_rate * gradients
        delta_gradient = -self.momentum * previous_velocity + (1 + self.momentum) * self.velocity[i]
        weights += delta_gradient
