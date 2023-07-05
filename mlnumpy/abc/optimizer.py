from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):
    @abstractmethod
    def __call__(
        self,
        weights: np.ndarray,
        gradients: np.ndarray,
    ) -> None:
        pass
