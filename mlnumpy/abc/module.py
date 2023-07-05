from abc import ABC, abstractmethod

import numpy as np


class Module(ABC):
    def __init__(self) -> None:
        self.__train_mode = True

    def __str__(self) -> str:
        return f"{self.__class__.__name__}()"

    @property
    def train_mode(self) -> bool:
        return self.__train_mode

    def train(self) -> None:
        self.__train_mode = True

    def eval(self) -> None:
        self.__train_mode = False

    @abstractmethod
    def __call__(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def backward(
        self,
        errors: np.ndarray,
    ) -> np.ndarray:
        pass
