import numpy as np

from mlnumpy.abc.optimizer import Optimizer


class Adam(Optimizer):
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m: dict[int, np.ndarray] = {}
        self.v: dict[int, np.ndarray] = {}
        self.t: dict[int, np.ndarray] = {}

    def __call__(
        self,
        weights: np.ndarray,
        gradients: np.ndarray,
    ) -> None:
        i = id(weights)
        self.m[i] = self.beta1 * self.m.get(i, 0) + (1 - self.beta1) * gradients
        self.v[i] = self.beta2 * self.v.get(i, 0) + (1 - self.beta2) * gradients**2

        self.t[i] = self.t.get(i, 0) + 1
        m_hat = self.m[i] / (1 - self.beta1 ** self.t[i])
        v_hat = self.v[i] / (1 - self.beta2 ** self.t[i])

        weights += self.learning_rate / (np.sqrt(v_hat) + self.epsilon) * m_hat
