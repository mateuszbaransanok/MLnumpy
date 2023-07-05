from abc import ABC, abstractmethod

import numpy as np

from mlnumpy.abc.activation import Activation


class Loss(ABC):
    @abstractmethod
    def __call__(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        pass

    @abstractmethod
    def validate(
        self,
        activation: Activation,
    ) -> bool:
        pass
